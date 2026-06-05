# VBD Cloth Simulation

GPU square-cloth simulation based on NVIDIA Warp and Vertex Block Descent.
The cloth is fixed at the two upper corners and naturally hangs under gravity.

## Environment

This project is managed with PDM.

```powershell
cd D:\code\VBD
pdm install
```

Check Warp and CUDA:

```powershell
pdm run python -c "import warp as wp; wp.init(); print(wp.__version__, wp.is_cuda_available())"
```

## Run

Default simulation:

```powershell
pdm run cloth
```

Small smoke test:

```powershell
pdm run python warp_vbd_cloth_no_contact_jml.py --resolution 16 --frames 5 --save-every 1
```

The default output directory is:

```text
D:\code\VBD\vbd_cloth_output
```

The output files are `.vtp` PolyData files and can be opened directly in ParaView.

The repository also includes `vbd_cloth_output.zip`, a packaged sample output
that can be extracted and opened in ParaView.

## Notes

- `.venv/`, `.warp_cache/`, and `vbd_cloth_output/` are generated locally and are not tracked.
- The VBD solve uses 9-color vertex grouping so vertices of the same color can be updated in parallel on the GPU.
