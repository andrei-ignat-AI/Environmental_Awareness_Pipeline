import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path

RIGHT_ARROW_KEY = 262
LEFT_ARROW_KEY = 263
VOXEL_SIZE = 0.05

def mujoco_to_open3d_vector(vector):
    """Inverse of the right-handed axis swap [x, -z, y] used in export."""
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([x, z, -y], dtype=np.float64)

def load_exported_sample(run_dir: Path, sample_rel_path: str, sample_name: str):
    export_dir = run_dir / "mujoco_exports" / sample_rel_path
    if not export_dir.exists():
        # Fallback in case paths are flat
        export_dir = run_dir / "mujoco_exports" / sample_name

    json_path = export_dir / f"{sample_name}_boxes.json"
    ply_path = export_dir / f"{sample_name}_voxels.ply"

    geometries = []

    # 1. Load Voxels
    if ply_path.exists():
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd and not pcd.is_empty():
            # Convert point cloud back to Open3D VoxelGrid for true cubic visualization
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
            geometries.append(voxel_grid)

    # 2. Load Convex Hulls & Bounding Boxes
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for geom in data.get("geoms", []):
            rgba = geom.get("rgba", [0.5, 0.5, 0.5, 1.0])
            color = np.array(rgba[:3], dtype=np.float64)

            if geom.get("geom_type") == "mesh":
                vertices_mj = np.array(geom.get("vertices_mujoco", []))
                faces = np.array(geom.get("faces", []))

                if len(vertices_mj) > 0:
                    vertices_o3d = np.array([mujoco_to_open3d_vector(v) for v in vertices_mj])
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(vertices_o3d)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    mesh.compute_vertex_normals()
                    mesh.paint_uniform_color(color)
                    
                    # Create wireframe visualization
                    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                    wireframe.paint_uniform_color(color)
                    geometries.append(wireframe)

            elif "center_open3d" in geom:
                center = np.array(geom["center_open3d"])
                extent = np.array(geom["extent_open3d"])
                rotation = np.array(geom["rotation_open3d"]).reshape(3, 3)

                box = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
                box.color = color
                geometries.append(box)
                
    return geometries

def run_viewer(run_dir: Path, start_index: int = 0, auto_spin: bool = False):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sample_results = manifest.get("sample_results", [])
    if not sample_results:
        print("No samples found in manifest.")
        return

    if start_index < 0 or start_index >= len(sample_results):
        print(f"Warning: Start index {start_index} out of bounds (0 to {len(sample_results)-1}). Defaulting to 0.")
        start_index = 0

    state = {"index": start_index, "geometries": [], "spinning": auto_spin}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Native Run Viewer: {run_dir.name}")

    def load_sample(index):
        for geom in state["geometries"]:
            vis.remove_geometry(geom, reset_bounding_box=False)

        sample_info = sample_results[index].get("sample", {})
        sample_name = sample_info.get("sample_name")
        sample_rel_path = sample_info.get("relative_path", sample_name)
        print(f"[{index+1}/{len(sample_results)}] Loading {sample_rel_path}...")

        geoms = load_exported_sample(run_dir, sample_rel_path, sample_name)
        state["index"] = index
        state["geometries"] = geoms
        for geom in geoms:
            vis.add_geometry(geom, reset_bounding_box=True)

    def on_right(vis): load_sample((state["index"] + 1) % len(sample_results)); return False
    def on_left(vis): load_sample((state["index"] - 1) % len(sample_results)); return False

    def animation_callback(vis):
        if state["spinning"]:
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            return True
        return False

    def on_space(vis):
        state["spinning"] = not state["spinning"]
        if state["spinning"]:
            vis.register_animation_callback(animation_callback)
        else:
            vis.register_animation_callback(None)
        return False

    if auto_spin:
        vis.register_animation_callback(animation_callback)

    load_sample(start_index)
    vis.register_key_callback(RIGHT_ARROW_KEY, on_right)
    vis.register_key_callback(LEFT_ARROW_KEY, on_left)
    vis.register_key_callback(32, on_space)  # 32 is the keycode for the Space bar
    print("\nControls:\n  Right Arrow : Next Sample\n  Left Arrow  : Previous Sample\n  Space       : Toggle Auto-Spin\n")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native Open3D viewer for exported noise runs.")
    parser.add_argument("run_id", help="The name of the run folder in outputs/runs/")
    parser.add_argument("--sample", type=int, default=0, help="Starting sample index (default: 0)")
    parser.add_argument("--spin", action="store_true", help="Automatically spin the camera on startup")
    args = parser.parse_args()
    run_viewer(Path(__file__).resolve().parents[1] / "outputs" / "runs" / args.run_id, start_index=args.sample, auto_spin=args.spin)
