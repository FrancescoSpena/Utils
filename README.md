# Visualize.py

Utility script to visualize `.ply` meshes, pose trajectories (4×4 `c2w` matrices), and/or point clouds with [**Rerun Viewer**](https://www.rerun.io/).  
It is designed for analyzing the results of **Point-SLAM** and **KAN-SLAM**.

---

## Environment

This script must be executed inside the Conda environment **`visualize-res`**.  
If you haven’t install it yet:

```bash
conda env create -f env.yaml
```

If you haven’t activated it yet:

```bash
conda activate visualize-res
```

Make sure the required packages are installed:

```bash
pip install open3d==0.17.0 rerun-sdk numpy
```

---

## Usage

```bash
python visualize.py --mesh PATH/mesh.ply [--traj PATH/traj.txt] [--cloud PATH/cloud.ply]
```

### Main arguments

- `--mesh` (required): `.ply` mesh file.
- `--traj` (optional): trajectory text file.  
  Each line must contain **16 floats** = 4×4 `c2w` matrix (row-major).  
  Empty lines or lines starting with `#` are ignored.
- `--cloud` (optional): final point cloud `.ply`.

To display all arguments:

```bash
python visualize.py -h
```

---

## Examples

- **Mesh only**
  ```bash
  python visualize.py --mesh data/room0_mesh.ply
  ```

- **Mesh + trajectory subsampled every 5 frames**
  ```bash
  python visualize.py --mesh data/room0_mesh.ply --traj logs/traj.txt --every 5
  ```

- **Mesh + point cloud with 2 cm downsample**
  ```bash
  python visualize.py --mesh data/room0_mesh.ply --cloud results/fused_cloud.ply --voxel_size 0.02
  ```

---

## Trajectory format

- **One line per frame**
- **16 floats separated by spaces** → 4×4 matrix in row-major order
- Example (identity matrix):
  ```
  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
  ```

By default, a **flip on Y and Z axes** is applied (compatibility with Point-SLAM and KAN-SLAM).  
If you don’t want this correction, modify the script by removing the `flip_axes` flag.

---

## Output

- Mesh, trajectory, and point cloud are logged to **Rerun Viewer**.
- The trajectory is shown as a colored line with:
  - Start point = green
  - End point = red

---

## Notes

- If you run on a server/headless machine and encounter X11/GUI issues (`GLFW Error: DISPLAY`), you can save the `.rrd` session and visualize it locally with Rerun by defining the out_rrd name from the flag.

---
