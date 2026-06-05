# -*- coding: utf-8 -*-
"""Warp GPU 布料仿真：两角固定的方形布料自然下垂。

本文件是一个可直接运行的示例程序，核心目标有三个：
1. 在 CPU 端生成方形布料网格、三角面片和弹簧拓扑。
2. 在 GPU 端用 NVIDIA Warp kernel 执行隐式时间积分和 VBD 求解。
3. 输出 ParaView 可查看的 .vtp 帧文件，便于检查布料形变和变量。

VBD(Vertex Block Descent) 的思想是：一次只优化一个顶点的 3 维位置块，
把相邻顶点暂时看成已知值，构造该顶点局部能量的一阶梯度和 3x3 Hessian，
然后求一个局部牛顿步。为了能在 GPU 上并行更新，顶点被分成 9 种颜色；
同一颜色的顶点不会通过本程序中的弹簧直接相连，因此可以同时更新。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


# 每个顶点最多记录的邻居弹簧数量。
# 当前网格包含结构弹簧、剪切弹簧和两跳弯曲弹簧，16 个槽位足够覆盖内部顶点。
MAX_NEIGHBORS = 16

# 3x3 顶点着色。因为存在横向/纵向两跳弯曲弹簧，普通 2x2 四色不够；
# 使用 3x3 九色后，同色顶点之间不会共享本程序生成的任何弹簧。
NUM_COLORS = 9
SCRIPT_DIR = Path(__file__).resolve().parent


# 预测步 kernel：执行隐式 Euler 的惯性预测。
# 对每个自由顶点，先根据上一帧位置、速度和重力得到预测位置 y；
# VBD 后续求解会在 y 附近寻找满足弹簧约束的隐式位置。
@wp.kernel
def predict_kernel(
    x: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inertial: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int32),
    pinned_x: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
):
    tid = wp.tid()

    if fixed[tid] != 0:
        # 固定点每个子步都强制回到钉住位置，速度清零。
        # 这样即使数值误差或输出读写导致位置发生偏移，也会被立即纠正。
        p = pinned_x[tid]
        x[tid] = p
        x_old[tid] = p
        inertial[tid] = p
        v[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        # y = x_n + h * v_n + h^2 * g。
        # 注意这里没有先显式积分到最终位置；y 只是隐式目标函数里的惯性中心。
        old = x[tid]
        x_old[tid] = old
        y = old + v[tid] * dt + gravity * (dt * dt) 
        inertial[tid] = y
        x[tid] = y


# VBD 单颜色更新 kernel。
# active_color 指定当前处理哪一组顶点；同色顶点可以并行更新。
# 每个线程只负责一个顶点的 3x3 局部问题：
#
# E_i(x_i) = inertia/2 * ||x_i - y_i||^2
#          + sum_j 0.5 * k_ij * (||x_i - x_j|| - L_ij)^2
#
# 其中 inertia = m / h^2，来自隐式 Euler 的惯性项。
@wp.kernel
def vbd_color_kernel(
    x: wp.array(dtype=wp.vec3),
    inertial: wp.array(dtype=wp.vec3),
    neighbors: wp.array(dtype=wp.int32),
    rest_lengths: wp.array(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    colors: wp.array(dtype=wp.int32),
    active_color: int,
    inertia: float,
    max_step: float,
):
    tid = wp.tid()

    if fixed[tid] != 0 or colors[tid] != active_color:
        return

    xi = x[tid]
    yi = inertial[tid]

    # 梯度初始化为惯性项：inertia * (x_i - y_i)。
    gx = inertia * (xi[0] - yi[0])
    gy = inertia * (xi[1] - yi[1])
    gz = inertia * (xi[2] - yi[2])

    # Hessian 初始化为惯性项的 3x3 对角矩阵。
    # 为了减少寄存器和计算量，这里只存对称矩阵的 6 个独立元素：
    # [h00 h01 h02]
    # [h01 h11 h12]
    # [h02 h12 h22]
    h00 = inertia
    h01 = 0.0
    h02 = 0.0
    h11 = inertia
    h12 = 0.0
    h22 = inertia

    base = tid * MAX_NEIGHBORS

    for slot in range(MAX_NEIGHBORS):
        n = neighbors[base + slot]

        if n >= 0:
            # 从扁平邻接表中读取一根弹簧：(tid, n)。
            # rest 是静止长度 L，k 是该弹簧的刚度。
            xj = x[n]
            rest = rest_lengths[base + slot]
            k = stiffness[base + slot]

            dx = xi - xj
            r = wp.length(dx)

            if r > 1.0e-7:
                # 单位方向 n = (x_i - x_j) / ||x_i - x_j||。
                inv_r = 1.0 / r
                nx = dx[0] * inv_r
                ny = dx[1] * inv_r
                nz = dx[2] * inv_r

                # 对弹簧能量 0.5*k*(r-L)^2 求梯度：
                # grad = k * (1 - L/r) * (x_i - x_j)。
                stretch = 1.0 - rest * inv_r

                gx += k * stretch * dx[0]
                gy += k * stretch * dx[1]
                gz += k * stretch * dx[2]

                # 弹簧 Hessian 的完整形式包含切向和法向分量。
                # 当弹簧被压缩时，精确 Hessian 可能非正定，导致牛顿步不稳定；
                # 这里把切向系数裁到非负，得到一个更稳的半正定近似。
                tangent = stretch
                if tangent < 0.0:
                    tangent = 0.0

                kt = k * tangent
                kn = k * (1.0 - tangent)

                # H = kt * I + kn * n*n^T，只累加对称矩阵的 6 个元素。
                h00 += kt + kn * nx * nx
                h01 += kn * nx * ny
                h02 += kn * nx * nz
                h11 += kt + kn * ny * ny
                h12 += kn * ny * nz
                h22 += kt + kn * nz * nz

    a = h00
    b = h01
    c = h02
    d = h11
    e = h12
    f = h22

    # 手写 3x3 对称矩阵求逆的伴随矩阵部分。
    # Warp kernel 中避免调用通用线性代数库，直接展开可以减少开销。
    cof00 = d * f - e * e
    cof01 = c * e - b * f
    cof02 = b * e - c * d
    cof11 = a * f - c * c
    cof12 = b * c - a * e
    cof22 = a * d - b * b

    det = a * cof00 + b * cof01 + c * cof02

    if det > 1.0e-10 or det < -1.0e-10:
        inv_det = 1.0 / det

        # 解 H * s = grad，局部牛顿更新为 x <- x - s。
        sx = (cof00 * gx + cof01 * gy + cof02 * gz) * inv_det
        sy = (cof01 * gx + cof11 * gy + cof12 * gz) * inv_det
        sz = (cof02 * gx + cof12 * gy + cof22 * gz) * inv_det

        step = wp.vec3(-sx, -sy, -sz)
        step_len = wp.length(step)

        if step_len > max_step:
            # 限制单次局部更新长度，避免低迭代数或极端参数下出现过冲。
            step = step * (max_step / step_len)

        x[tid] = xi + step


# 收尾 kernel：用最终位置回算速度。
# VBD 求出的 x 是隐式积分后的新位置；速度用 (x_{n+1}-x_n)/h 得到，
# 再乘一个轻微阻尼，减少布料长时间振荡。
@wp.kernel
def finalize_kernel(
    x: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int32),
    pinned_x: wp.array(dtype=wp.vec3),
    dt: float,
    damping: float,
):
    tid = wp.tid()

    if fixed[tid] != 0:
        # 固定点速度始终为 0，位置始终等于钉住位置。
        p = pinned_x[tid]
        x[tid] = p
        v[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v[tid] = ((x[tid] - x_old[tid]) / dt) * damping


@dataclass
class ClothData:
    """CPU 端构建好的布料数据包。

    这些数组会在 simulate() 中上传到 Warp 设备端。faces 只用于输出网格，
    弹簧邻接表才是 VBD 求解真正使用的拓扑。
    """

    # 顶点初始位置，同时也是位移输出 displacement 的参考形状。
    positions: np.ndarray
    # 固定点目标位置；非固定点也填入初始位置，便于统一上传。
    pinned_positions: np.ndarray
    # fixed[i] = 1 表示顶点 i 被钉住，0 表示自由顶点。
    fixed: np.ndarray
    # VBD 并行着色编号，范围为 0..8。
    colors: np.ndarray
    # 扁平邻接表：第 i 个顶点的邻居在 neighbors[i*MAX_NEIGHBORS:(i+1)*MAX_NEIGHBORS]。
    neighbors: np.ndarray
    # 与 neighbors 对齐的弹簧静止长度。
    rest_lengths: np.ndarray
    # 与 neighbors 对齐的弹簧刚度。
    stiffness: np.ndarray
    # 三角面片索引，只用于 VTP 输出。
    faces: np.ndarray
    # 每个顶点的集中质量，用于 inertia = m / h^2。
    vertex_mass: float
    # 网格间距，用于弹簧静止长度和牛顿步长限制。
    dx: float


def vertex_index(ix: int, iy: int, nx: int) -> int:
    """把二维网格坐标映射为一维顶点编号。"""
    return iy * nx + ix


def add_spring(
    a: int,
    b: int,
    rest: float,
    k: float,
    neighbors: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: np.ndarray,
    counts: np.ndarray,
):
    """向邻接表中加入一根无向弹簧。

    VBD 更新顶点 i 时需要知道所有相邻顶点 j，因此一根物理弹簧要同时写入
    a 的邻接表和 b 的邻接表。counts 记录每个顶点已经使用了多少个槽位。
    """

    for src, dst in ((a, b), (b, a)):
        slot = counts[src]
        if slot >= MAX_NEIGHBORS:
            raise RuntimeError(f"MAX_NEIGHBORS={MAX_NEIGHBORS} is too small for vertex {src}")

        neighbors[src, slot] = dst
        rest_lengths[src, slot] = rest
        stiffness[src, slot] = k
        counts[src] += 1


def build_square_cloth(
    resolution: int,
    size: float,
    density: float,
    stretch_stiffness: float,
    shear_stiffness: float,
    bend_stiffness: float,
) -> ClothData:
    """构建方形布料的几何、弹簧和固定点。

    布料位于 x-y 平面附近，上边缘 y=0，向下为负 y。
    左上角和右上角固定，其余顶点自由运动。初始 z 坐标加入很小扰动，
    让布料下垂时更容易产生自然褶皱，而不是完全保持平面。
    """

    if resolution < 3:
        raise ValueError("resolution must be at least 3")

    nx = resolution
    ny = resolution
    n_vertices = nx * ny
    dx = size / float(resolution - 1)

    positions = np.zeros((n_vertices, 3), dtype=np.float32)
    pinned_positions = np.zeros_like(positions)
    fixed = np.zeros(n_vertices, dtype=np.int32)
    colors = np.zeros(n_vertices, dtype=np.int32)

    # 生成规则方形网格顶点。
    # x 横向覆盖 [-size/2, size/2]，y 从 0 向下排布。
    for iy in range(ny):
        for ix in range(nx):
            idx = vertex_index(ix, iy, nx)
            x = (float(ix) / float(nx - 1) - 0.5) * size
            y = -float(iy) * dx
            z = 0.008 * dx * np.sin(1.73 * ix + 2.41 * iy)
            positions[idx] = (x, y, z)
            pinned_positions[idx] = positions[idx]
            colors[idx] = (ix % 3) + 3 * (iy % 3)

    # 固定布料上边缘的两个角点。
    # 只有这两个顶点 fixed=1，其余顶点在重力下自然垂落。
    top_left = vertex_index(0, 0, nx)
    top_right = vertex_index(nx - 1, 0, nx)
    fixed[top_left] = 1
    fixed[top_right] = 1
    positions[top_left, 2] = 0.0
    positions[top_right, 2] = 0.0
    pinned_positions[top_left] = positions[top_left]
    pinned_positions[top_right] = positions[top_right]

    neighbors = np.full((n_vertices, MAX_NEIGHBORS), -1, dtype=np.int32)
    rest_lengths = np.zeros((n_vertices, MAX_NEIGHBORS), dtype=np.float32)
    stiffness = np.zeros((n_vertices, MAX_NEIGHBORS), dtype=np.float32)
    counts = np.zeros(n_vertices, dtype=np.int32)

    # 为网格加入三类弹簧：
    # 1. 结构弹簧：水平/竖直相邻点，控制主要拉伸。
    # 2. 剪切弹簧：每个方格的两条对角线，控制斜向变形。
    # 3. 弯曲弹簧：跨两个网格间距的水平/竖直弹簧，近似控制弯曲刚度。
    for iy in range(ny):
        for ix in range(nx):
            i = vertex_index(ix, iy, nx)

            if ix + 1 < nx:
                add_spring(i, vertex_index(ix + 1, iy, nx), dx, stretch_stiffness, neighbors, rest_lengths, stiffness, counts)
            if iy + 1 < ny:
                add_spring(i, vertex_index(ix, iy + 1, nx), dx, stretch_stiffness, neighbors, rest_lengths, stiffness, counts)

            if ix + 1 < nx and iy + 1 < ny:
                add_spring(i, vertex_index(ix + 1, iy + 1, nx), np.sqrt(2.0) * dx, shear_stiffness, neighbors, rest_lengths, stiffness, counts)
            if ix + 1 < nx and iy - 1 >= 0:
                add_spring(i, vertex_index(ix + 1, iy - 1, nx), np.sqrt(2.0) * dx, shear_stiffness, neighbors, rest_lengths, stiffness, counts)

            if ix + 2 < nx:
                add_spring(i, vertex_index(ix + 2, iy, nx), 2.0 * dx, bend_stiffness, neighbors, rest_lengths, stiffness, counts)
            if iy + 2 < ny:
                add_spring(i, vertex_index(ix, iy + 2, nx), 2.0 * dx, bend_stiffness, neighbors, rest_lengths, stiffness, counts)

    # VTP 输出需要三角面片。
    # 每个四边形网格单元拆成两个三角形。
    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = vertex_index(ix, iy, nx)
            b = vertex_index(ix + 1, iy, nx)
            c = vertex_index(ix, iy + 1, nx)
            d = vertex_index(ix + 1, iy + 1, nx)
            faces.append((a, c, b))
            faces.append((b, c, d))

    area = size * size
    # 简单使用均匀集中质量。更精细的实现可以按三角形面积分配顶点质量。
    vertex_mass = density * area / float(n_vertices)

    return ClothData(
        positions=positions,
        pinned_positions=pinned_positions,
        fixed=fixed,
        colors=colors,
        neighbors=neighbors.reshape(-1),
        rest_lengths=rest_lengths.reshape(-1),
        stiffness=stiffness.reshape(-1),
        faces=np.asarray(faces, dtype=np.int32),
        vertex_mass=vertex_mass,
        dx=dx,
    )


def _format_float_array(values: np.ndarray, components: int = 1) -> str:
    """把浮点数组格式化成 VTK XML ASCII DataArray 需要的文本。"""

    flat = np.asarray(values, dtype=np.float32).reshape(-1, components)
    return "\n".join(" ".join(f"{x:.7g}" for x in row) for row in flat)


def _format_int_array(values: np.ndarray, components: int = 1) -> str:
    """把整数数组格式化成 VTK XML ASCII DataArray 需要的文本。"""

    flat = np.asarray(values, dtype=np.int32).reshape(-1, components)
    return "\n".join(" ".join(str(int(x)) for x in row) for row in flat)


def write_vtp(
    path: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_positions: np.ndarray,
    faces: np.ndarray,
    fixed: np.ndarray,
    colors: np.ndarray,
):
    """写出一帧 VTP PolyData，供 ParaView 读取。

    VTP 是 VTK XML PolyData 文件。这里把布料保存为三角面片网格，并额外写入
    点数据数组：
    - velocity：顶点速度向量。
    - speed：速度长度，方便在 ParaView 里直接按速度着色。
    - displacement：相对初始形状的位移。
    - fixed：是否为固定顶点。
    - vbd_color：VBD 的 9 色并行分组编号。

    文件使用 ASCII 格式，体积比二进制大一点，但便于调试和查看。
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = positions.shape[0]
    n_faces = faces.shape[0]
    displacement = positions - rest_positions
    speed = np.linalg.norm(velocities, axis=1).astype(np.float32)
    # VTK 的 Polys 由 connectivity 和 offsets 两个数组描述：
    # connectivity 是所有三角形顶点索引顺序拼接；
    # offsets 表示每个单元在 connectivity 中结束的位置。三角形每个单元 3 个点。
    offsets = np.arange(3, 3 * n_faces + 1, 3, dtype=np.int32)

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfPolys="{n_faces}">\n')
        f.write('      <PointData Scalars="speed" Vectors="velocity">\n')
        f.write('        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n')
        f.write(_format_float_array(velocities, 3))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="speed" format="ascii">\n')
        f.write(_format_float_array(speed))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="displacement" NumberOfComponents="3" format="ascii">\n')
        f.write(_format_float_array(displacement, 3))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="fixed" format="ascii">\n')
        f.write(_format_int_array(fixed))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="vbd_color" format="ascii">\n')
        f.write(_format_int_array(colors))
        f.write("\n        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("      <CellData>\n")
        f.write("      </CellData>\n")
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">\n')
        f.write(_format_float_array(positions, 3))
        f.write("\n        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Polys>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(_format_int_array(faces.reshape(-1)))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(_format_int_array(offsets))
        f.write("\n        </DataArray>\n")
        f.write("      </Polys>\n")
        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def write_pvd(path: Path, frames: list[tuple[int, float, str]]):
    """Write a ParaView time-series index for the saved VTP frames."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for _frame, time, filename in frames:
            f.write(f'    <DataSet timestep="{time:.7g}" group="" part="0" file="{filename}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


def simulate(args):
    """运行完整仿真流程。

    主循环结构为：
    1. 构建 CPU 端布料数据。
    2. 上传到 Warp 设备数组。
    3. 每帧执行若干 substep。
    4. 每个 substep 执行预测、VBD 颜色迭代和速度回算。
    5. 按 save_every 写出 VTP。
    """

    if wp.config.kernel_cache_dir is None:
        wp.config.kernel_cache_dir = str(SCRIPT_DIR / ".warp_cache")

    wp.init()

    # 先在 CPU 端建立规则网格、弹簧邻接表、固定点和三角面片。
    cloth = build_square_cloth(
        resolution=args.resolution,
        size=args.size,
        density=args.density,
        stretch_stiffness=args.stretch_stiffness,
        shear_stiffness=args.shear_stiffness,
        bend_stiffness=args.bend_stiffness,
    )

    device = args.device
    n_vertices = cloth.positions.shape[0]

    # 将 CPU numpy 数组上传到 Warp 设备端。
    # x 是当前顶点位置，x_old 保存子步开始时的位置，v 是速度。
    # inertial 保存预测位置 y，后续 VBD 能量里的惯性项会使用它。
    x = wp.from_numpy(cloth.positions, dtype=wp.vec3, device=device)
    x_old = wp.from_numpy(cloth.positions.copy(), dtype=wp.vec3, device=device)
    v = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    inertial = wp.from_numpy(cloth.positions.copy(), dtype=wp.vec3, device=device)

    # 固定点、颜色和弹簧拓扑在仿真过程中不变，只需上传一次。
    pinned_x = wp.from_numpy(cloth.pinned_positions, dtype=wp.vec3, device=device)
    fixed = wp.from_numpy(cloth.fixed, dtype=wp.int32, device=device)
    colors = wp.from_numpy(cloth.colors, dtype=wp.int32, device=device)
    neighbors = wp.from_numpy(cloth.neighbors, dtype=wp.int32, device=device)
    rest_lengths = wp.from_numpy(cloth.rest_lengths, dtype=wp.float32, device=device)
    stiffness = wp.from_numpy(cloth.stiffness, dtype=wp.float32, device=device)

    out_dir = Path(args.out_dir)

    # frame_dt 是输出帧时间间隔；sub_dt 是物理求解子步时间间隔。
    # inertia = m / h^2 是隐式 Euler 目标函数中的质量项权重。
    frame_dt = 1.0 / args.fps
    sub_dt = frame_dt / float(args.substeps)
    inertia = cloth.vertex_mass / (sub_dt * sub_dt)
    max_step = args.max_step_scale * cloth.dx
    gravity = wp.vec3(0.0, args.gravity, 0.0)
    saved_vtk_frames: list[tuple[int, float, str]] = []

    # VTP 输出从第 0 帧开始记录。
    vtk_name = "frame_0000.vtp"
    zero_velocity = np.zeros_like(cloth.positions)
    write_vtp(
        out_dir / vtk_name,
        cloth.positions,
        zero_velocity,
        cloth.positions,
        cloth.faces,
        cloth.fixed,
        cloth.colors,
    )
    saved_vtk_frames.append((0, 0.0, vtk_name))

    for frame in range(1, args.frames + 1):
        for _substep in range(args.substeps):
            # 预测步：计算隐式 Euler 的惯性目标位置 y。
            wp.launch(
                predict_kernel,
                dim=n_vertices,
                inputs=[x, x_old, v, inertial, fixed, pinned_x, gravity, sub_dt],
                device=device,
            )

            # VBD 迭代：每次迭代依次扫 9 个颜色。
            # 同一颜色顶点之间没有直接弹簧连接，因此该颜色内可以 GPU 并行更新。
            for _iter in range(args.vbd_iters):
                for color in range(NUM_COLORS):
                    wp.launch(
                        vbd_color_kernel,
                        dim=n_vertices,
                        inputs=[
                            x,
                            inertial,
                            neighbors,
                            rest_lengths,
                            stiffness,
                            fixed,
                            colors,
                            color,
                            inertia,
                            max_step,
                        ],
                        device=device,
                    )

            # 用本子步开始和结束的位置差回算速度，并施加阻尼。
            wp.launch(
                finalize_kernel,
                dim=n_vertices,
                inputs=[x, x_old, v, fixed, pinned_x, sub_dt, args.damping],
                device=device,
            )

        if frame % args.save_every == 0 or frame == args.frames:
            # 只有保存输出时才把 GPU 位置/速度拷回 CPU，减少不必要的数据传输。
            positions = x.numpy()
            velocities = v.numpy()
            vtk_name = f"frame_{frame:04d}.vtp"

            write_vtp(
                out_dir / vtk_name,
                positions,
                velocities,
                cloth.positions,
                cloth.faces,
                cloth.fixed,
                cloth.colors,
            )
            saved_vtk_frames.append((frame, frame * frame_dt, vtk_name))

            print(f"saved frame {frame:04d}")

    pvd_path = out_dir / "cloth.pvd"
    write_pvd(pvd_path, saved_vtk_frames)
    print(f"wrote ParaView VTP frames to: {out_dir.resolve()}")
    print(f"wrote ParaView cloth time series: {pvd_path.resolve()}")


def parse_args():
    """解析命令行参数。

    VSCode 的 .vscode/launch.json 也是通过这些参数启动脚本。
    修改分辨率、帧数、刚度、输出目录等都可以直接改这里的默认值，
    或者在命令行/VSCode 配置中传入覆盖值。
    """

    parser = argparse.ArgumentParser(
        description="GPU square-cloth simulation with Warp and Vertex Block Descent.",
    )
    parser.add_argument("--resolution", type=int, default=41, help="方形布料边上不知41个点")
    parser.add_argument("--size", type=float, default=2.0, help="square cloth side length")

    parser.add_argument("--frames", type=int, default=40, help="number of output frames to simulate")
    parser.add_argument("--fps", type=float, default=60.0, help="simulation frame rate")
    parser.add_argument("--substeps", type=int, default=1, help="substeps per frame")

    parser.add_argument("--vbd-iters", type=int, default=18, help="VBD color-sweep iterations per substep")
    parser.add_argument("--density", type=float, default=0.18, help="surface mass density")
    parser.add_argument("--stretch-stiffness", type=float, default=1500.0, help="structural spring stiffness")
    parser.add_argument("--shear-stiffness", type=float, default=1000.0, help="diagonal shear spring stiffness")
    parser.add_argument("--bend-stiffness", type=float, default=90.0, help="two-hop bending spring stiffness")
    parser.add_argument("--damping", type=float, default=0.992, help="velocity damping after each substep")
    parser.add_argument("--gravity", type=float, default=-9.81, help="Y-axis gravity acceleration")
    parser.add_argument("--max-step-scale", type=float, default=0.6, help="limits one local VBD step to this multiple of grid spacing")
    parser.add_argument("--device", default="cuda:0", help="Warp device, e.g. cuda:0 or cpu")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "vbd_cloth_output_nocontact"), help="directory for ParaView VTP frames")
    parser.add_argument("--save-every", type=int, default=5, help="VTP save interval in frames")
    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
