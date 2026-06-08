import argparse
import colorsys
import copy
import os
import sys
from collections import deque

import json
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError
from yourdfpy import URDF

from . import azurion_dataset


# ---------------------------------Definitions of Variables---------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_ASSET_ROOT = os.path.join(PROJECT_ROOT, "assets", "Robotarm")
ROBOT_HULLS_FOLDER = os.path.join(ROBOT_ASSET_ROOT, "hulls")
ROBOT_URDF_PATH = os.path.join(ROBOT_ASSET_ROOT, "FlexArmStudents.urdf")
default_voxel_size = 0.05  # Keep the Python grid aligned with the current 5 cm Unity voxel export.
default_room_dimensions = (9.9, 2.7, 5.9)  # Unity room interior dimensions: x, y, z.
default_wall_thickness = 0.01  # Default wall thickness for visualization
DepthImageFolder = os.environ.get(
    "AZURION_CAPTURE_FOLDER",
    azurion_dataset.DEFAULT_CAPTURE_ROOT,
)  # Folder containing sample_* depth captures
X_OFFSET = 0.0
Y_OFFSET = 2.5999999046325685
Z_OFFSET = 0.0
JSON_NAMING_CONVENTION = {
    "Carriage": "Long",
    "HorBeam": "Z1Rot",
    "VerBeam": "Z2Rot",
    "Sleeve": "Prop",
    "CArc": "CArc",
}


def draft_env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


DISPLAY_OCCLUSION_GRID = False
SHOW_ROBOT_HULL_OVERLAY = False
SHOW_COMPONENT_BOXES = False
SHOW_COMPONENT_CONVEX_HULLS = True
SHOW_COMPONENT_VOXELS = True
SHOW_SIDE_RGB_IMAGE = True
SHOW_FULL_VOXEL_GRID = False
SHOW_BLIND_SPOTS = True
REMOVE_TABLE_IN_DECOMPOSITION = draft_env_bool("AZURION_REMOVE_TABLE", True)
INCLUDE_TABLE_IN_DECOMPOSITION = not REMOVE_TABLE_IN_DECOMPOSITION
MAX_COMPONENTS_TO_DRAW = 80
MIN_COMPONENT_VOXELS = 25
COMPONENT_CONNECTIVITY = 6
OBJECT_BOX_MARGIN_M = 0.10
CONVEX_HULL_MARGIN_M = 0.05
SPLIT_OBJECTS_INTO_SMALLER_BOXES = True
SPLIT_MAX_BOX_EDGE_M = 0.25
SPLIT_MIN_VOXELS_PER_BOX = 12
SPLIT_MAX_RECURSION_DEPTH = 10
SPLIT_MAX_BOXES_PER_OBJECT = 25
DEFAULT_SAMPLE_INDEX = 0
OPEN_INTERACTIVE_VIEW = os.environ.get("DECOMPOSITION_DRAFT_NO_VIEW") != "1"
DRAFT_USE_SAFE_DEPTH_PROJECTION = False
DRAFT_DEPTH_PIXEL_STRIDE = 1
REQUIRE_STABLE_OPEN3D_RUNTIME = False
REMOVE_ROOM_BOUNDARY_VOXELS = True
ROOM_BOUNDARY_MARGIN_M = 0.16
FLOOR_REMOVAL_HEIGHT_M = 0.03
CEILING_REMOVAL_MARGIN_M = 0.03
TABLE_REMOVAL_MARGIN_M = 0.05
REMOVE_TABLE_EDGE_COMPONENTS = True
TABLE_EDGE_COMPONENT_MARGIN_M = 0.13
TABLE_EDGE_COMPONENT_MAX_VOXELS = 400
REMOVE_TABLE_LOW_SUPPORT_CLEANUP = True
TABLE_LOW_SUPPORT_MARGIN_M = 0.08
TABLE_LOW_SUPPORT_HEIGHT_M = 0.18
TABLE_FIXTURE_EXPAND_VOXELS = 1
TABLE_FIXTURE_PROTECT_PROPS_EXPAND_VOXELS = 1
MERGE_OCCLUSIONS_WITH_OBJECTS = False
INCLUDE_DETACHED_OCCLUSION_ZONES = True
BUILD_OCCLUSION_ZONE_BOXES = True
SHOW_OCCLUSION_ZONE_BOXES = True
SHOW_PHANTOM_FREE_SPACE_DIAGNOSTIC = False
OCCLUSION_COLOR = np.array([0.48, 0.27, 0.11], dtype=np.float64)
BLIND_SPOT_COLOR = np.array([0.15, 0.25, 0.45], dtype=np.float64)
PHANTOM_COLOR = np.array([1.0, 0.0, 0.0], dtype=np.float64)
OCCLUSION_BOX_MARGIN_M = 0.05
OCCLUSION_ATTACHMENT_RADIUS_VOXELS = 1
OCCLUSION_MIN_COMPONENT_VOXELS = 25
OCCLUSION_MAX_COMPONENT_VOXELS = 100000
OCCLUSION_SURFACE_CLEARANCE_VOXELS = 1
OCCLUSION_MAX_BOXES_PER_ZONE = 4
FILL_ONE_VOXEL_GAPS = True
GAP_FILL_ITERATIONS = 1
REPORT_BOX_TILT_DIAGNOSTICS = True
BOX_TILT_NOTE_DEGREES = 8.0
RIGHT_ARROW_KEY = 262
SIDE_RGB_FILE_NAME = "cam_side_rgb.png"


def grid_count_for_length(length, resolution):
    # unity uses ceil(room size / voxel size); the small epsilon handles float export noise.
    return max(1, int(np.ceil((float(length) - 1e-6) / float(resolution))))



# ---------------------------------Class Definitions---------------------------------


class RobotModel:
    def __init__(self, hulls_folder=None, urdf_path=None):
        self.hulls_folder = hulls_folder or ROBOT_HULLS_FOLDER
        self.urdf_path = urdf_path or ROBOT_URDF_PATH
        self.parts = {}
        self._load_all_hulls()
        self.base_parts = {name: copy.deepcopy(mesh) for name, mesh in self.parts.items()}
        self.urdf_model = URDF.load(self.urdf_path)
        self.urdf_model.update_cfg(self.urdf_model.zero_cfg)
        self.zero_link_transforms = {
            link_name: np.asarray(self.urdf_model.get_transform(link_name), dtype=np.float64)
            for link_name in self.urdf_model.link_map.keys()
        }

    def _load_all_hulls(self):
        for filename in os.listdir(self.hulls_folder):
            if filename.endswith(".stl"):
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.hulls_folder, filename)

                mesh = o3d.io.read_triangle_mesh(path)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.6, 0.6, 0.7])

                self.parts[name] = mesh

    def get_all_meshes(self):
        return list(self.parts.values())

    def transform_part(self, part_name, matrix):
        if part_name in self.parts:
            self.parts[part_name].transform(matrix)

    def transform_entire_robot(self, matrix):
        for name in self.parts:
            self.transform_part(name, matrix)

    def reset_to_base_pose(self):
        self.parts = {name: copy.deepcopy(mesh) for name, mesh in self.base_parts.items()}

    def _parts_for_link(self, link_name):
        return [part_name for part_name in self.parts if part_name.startswith(link_name)]

    def apply_urdf_fk_pose(self, joint_values=None, world_transform=None):
        joint_values = joint_values or {}
        world_transform = np.eye(4) if world_transform is None else np.asarray(world_transform, dtype=np.float64)

        self.urdf_model.update_cfg(joint_values)

        self.reset_to_base_pose()

        for link_name in self.urdf_model.link_map.keys():
            matching_parts = self._parts_for_link(link_name)
            if not matching_parts:
                continue

            link_transform = np.asarray(self.urdf_model.get_transform(link_name), dtype=np.float64)
            delta_transform = link_transform @ np.linalg.inv(self.zero_link_transforms[link_name])
            full_transform = world_transform @ delta_transform
            for part_name in matching_parts:
                self.parts[part_name].transform(full_transform)





class RoomPointCloud():
    def __init__(self):
        self.point_cloud = o3d.geometry.PointCloud()

    def apply_noise(self, depth_data, sigma = 0.1, dropout_pixel=0.01):
        """Applies Gaussian noise and random dropout to the depth data."""
        noisy_depth = depth_data.copy()

        # Apply Gaussian noise
        noise = np.random.normal(0, sigma, size=depth_data.shape)
        noisy_depth += noise

        # Apply random dropout
        dropout_mask = np.random.rand(*depth_data.shape) < dropout_pixel
        noisy_depth[dropout_mask] = 0

        return noisy_depth

    def remove_noise_2d(self, depth_data, sigma=0.1):
        """Applies a simple Gaussian filter to reduce noise in the depth data."""
        from scipy.ndimage import median_filter
        mask  = (depth_data > 0)  # Only filter valid depth pixels

        # First apply a median filter to remove salt-and-pepper noise
        filtered = median_filter(depth_data, size=3)
        return np.where(mask, filtered, 0)

    def remove_noise_3d(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """Removes noise from the point cloud using statistical outlier removal."""
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd.select_by_index(ind)

    def add_DepthImage(self, image_path, intrinsic, extrinsic=np.eye(4),add_noise=True):
        """Converts depth image to PCD and adds it to the room using camera extrinsics."""
        # Unity .raw files are often binary float32 arrays
        width, height = intrinsic.width, intrinsic.height
        near_clip, far_clip = azurion_dataset.depth_clip_for_raw_path(image_path)
        print("near_clip:", near_clip, "far_clip:", far_clip)
        depth_data = np.fromfile(image_path, dtype=np.float32).reshape((height, width))

        #depth_data[~azurion_dataset.valid_depth_mask(depth_data, near_clip, far_clip)] = 0.0 Doesn't work

        if add_noise:
            depth_data = self.apply_noise(depth_data, sigma=0.02, dropout_pixel=0.01)
            depth_data[~azurion_dataset.valid_depth_mask(depth_data, near_clip, far_clip)] = 0.0

        # Unity GetPixels order is bottom-to-top; Open3D expects top-to-bottom
        depth_data = np.ascontiguousarray(np.flipud(depth_data))
        depth_data[depth_data > far_clip-azurion_dataset.FAR_CLIP_EPSILON] = 0.0        # correct line for depth truncation with a small epsilon to prevent Open3D from discarding valid points at the far plane.
        depth_o3d = o3d.geometry.Image(depth_data)

        # --- COORDINATE SYSTEM CONVERSION ---
        # 1. Fix the Camera: Flip Local Y and Z axes (Rows 1 and 2)
        ext = np.array(extrinsic, dtype=np.float64)
        ext[1, :] *= -1
        ext[2, :] *= -1

        # 2. Fix the World: Flip Global Z axis (Column 2) to stop the "X" crossover
        ext[:, 2] *= -1

        # Force float64 and memory contiguity to prevent Open3D Segfaults
        extrinsic_clean = np.ascontiguousarray(ext, dtype=np.float64)
        
        print("far_clip after epsilon adjustment:", far_clip - azurion_dataset.FAR_CLIP_EPSILON)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            intrinsic,
            extrinsic=extrinsic_clean,
            depth_scale=1.0,
            depth_trunc=max(near_clip, far_clip - azurion_dataset.FAR_CLIP_EPSILON),
        )
        self.point_cloud += pcd

    def crop_point_cloud(self, x_bound,y_bound,z_bound,margin=0.5):
        """Crops the point cloud to fit within the defined room dimensions plus a margin."""
        if self.point_cloud.is_empty():
            print("Error: Point cloud is empty. Cannot crop.")
            return

        # Define the bounding box for cropping
        min_bound = np.array([x_bound[0] - margin,
                              y_bound[0] - margin,
                              z_bound[0] - margin])
        max_bound = np.array([x_bound[1] + margin,
                              y_bound[1] + margin,
                              z_bound[1] + margin])

        # Crop the point cloud using the defined bounding box
        cropped_pcd = self.point_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

        # Update the point cloud with the cropped version
        self.point_cloud = cropped_pcd

    def add_point_cloud(self, pcd):
        self.point_cloud += pcd

    def color_point_cloud(self, color):
        """Colors the entire point cloud with a single color."""
        if self.point_cloud.is_empty():
            print("Error: Point cloud is empty. Cannot color.")
            return
        self.point_cloud.paint_uniform_color(color)

    def clear(self):
        self.point_cloud.clear()

    def visualize(self):
        if self.point_cloud.is_empty():
            print("Error: Point cloud is empty. Check if the sample directory contains valid .raw files.")
            return
        self.point_cloud.estimate_normals()  # Required for lighting
        # Use the modern web/PBR viewer
        o3d.visualization.draw_geometries([self.point_cloud], window_name="Room Visualization (Point Cloud)")

    def filter_robot_points(self, robot_model, make_red=True, extra_margin_m=0.0):
        if self.point_cloud.is_empty() or not robot_model.parts:
            return


        scene = o3d.t.geometry.RaycastingScene()

        for name, mesh in robot_model.parts.items():
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene.add_triangles(t_mesh)

        query_points = o3d.core.Tensor(np.asarray(self.point_cloud.points), dtype=o3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query_points)
        is_robot_mask = occupancy.numpy().astype(bool)

        # Expand the robot exclusion zone by a user-defined margin around the hull.
        if extra_margin_m > 0.0:
            signed_distance = scene.compute_signed_distance(query_points).numpy()
            is_robot_mask = np.logical_or(is_robot_mask, signed_distance <= extra_margin_m)

        points_count = len(self.point_cloud.points)
        if make_red:
            colors = np.asarray(self.point_cloud.colors)
            if colors.size == 0:
                colors = np.tile([0.5, 0.5, 0.5], (points_count, 1))
            colors[is_robot_mask] = [1.0, 0.0, 0.0]
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        else:
            self.point_cloud = self.point_cloud.select_by_index(np.where(~is_robot_mask)[0])

        print(
            f"Filtering complete: {np.sum(is_robot_mask)} robot points detected "
            f"(margin: {extra_margin_m:.3f} m)."
        )






class Dataset():

    def __init__(self, folder_path=DepthImageFolder):
        self.folder_path = folder_path
        self.samples = azurion_dataset.discover_samples(folder_path)
        self.sample_paths = [sample.path for sample in self.samples]
        self.List_of_Samples = [sample.relative_path for sample in self.samples]

    def get_random_samplepath(self):
        """Returns the path for a random sample from the dataset."""
        if not self.sample_paths:
            return None
        full_path=np.random.choice(self.sample_paths)

        return full_path

    def get_samplepath(self,number):
        """Returns the path for a specific sample from the dataset."""
        if not self.sample_paths:
            return None
        if 0 <= number < len(self.sample_paths):
            return self.sample_paths[number]
        else:
            print("Invalid sample number.")
            return None

    def get_all_samplepaths(self):
        """Returns a list of all sample paths in the dataset."""
        return list(self.sample_paths)

    def _sample_sort_key(self, sample_path):
        sample_name = os.path.basename(sample_path)
        try:
            return (0, int(sample_name.split("_", 1)[1]))
        except (IndexError, ValueError):
            return (1, os.path.relpath(sample_path, self.folder_path))

    def get_data(self, path):
        """Returns the depth image paths, intrinsics, and extrinsics for a given sample path."""
        if path is None:
            print("No sample path provided.")
            return [], [], []

        try:
            cameras = azurion_dataset.load_depth_cameras(path)
        except (FileNotFoundError, KeyError) as exc:
            print(f"Camera metadata not available for {path}: {exc}")
            return [], [], []

        return (
            [camera.depth_path for camera in cameras],
            [camera.intrinsic for camera in cameras],
            [camera.extrinsic for camera in cameras],
        )

    def get_robot_pose(self, path):
        """Returns the robot's pose (position and orientation) for a given sample path."""
        if path is None:
            print("No sample path provided.")
            return None, None

        canonical_pose_path = os.path.join(path, "robot_state.json")
        if os.path.exists(canonical_pose_path):
            with open(canonical_pose_path, 'r') as f:
                metadata = json.load(f)

            joints_by_name = {item.get("jointName"): item for item in metadata.get("joints", [])}
            carriage = joints_by_name.get("Long")
            if carriage is None:
                print(f"Canonical robot state has no Long joint at {canonical_pose_path}")
                return None, None

            world_position = carriage["worldPosition"]
            robot_position = np.array([
                world_position["x"] - X_OFFSET,
                world_position["y"] - Y_OFFSET,
                world_position["z"] - Z_OFFSET
            ])

            joint_rotation = {}
            for mapped_name in JSON_NAMING_CONVENTION.values():
                joint = joints_by_name.get(mapped_name)
                if joint is None:
                    continue
                # keep the old robot-removal convention: place the robot from the carriage transform,
                # then use revolute jointPosition values for the articulated links.
                joint_rotation[mapped_name] = 0 if mapped_name == "Long" else joint.get("jointPosition", 0)

            return robot_position, joint_rotation

        pose_path = os.path.join(path, "robot_pose.json")
        if not os.path.exists(pose_path):
            print(f"Robot pose not found at {pose_path}")
            return None, None

        with open(pose_path, 'r') as f:
            metadata = json.load(f)


        # Retrieve the robot's position from the metadata for the 'carriage' joint
        # and convert it to a numpy array [x, y, z]
        carriage = next((item for item in metadata["joints"] if item.get("name") == "Carriage"), None)
        if carriage is None:
            print(f"Legacy robot pose has no Carriage joint at {pose_path}")
            return None, None

        robot_position = np.array([
            carriage["worldPosition"]["x"]-X_OFFSET,  # keep the legacy Open3D robot alignment.
            carriage["worldPosition"]["y"]-Y_OFFSET,
            carriage["worldPosition"]["z"]-Z_OFFSET
        ])

        #retrieve joints orientations
        joint_rotation={}
        for item in metadata["joints"]:
            joint_name = item["name"]
            if joint_name in JSON_NAMING_CONVENTION:
                mapped_name = JSON_NAMING_CONVENTION[joint_name]
                if joint_name == "Carriage":
                    joint_rotation[mapped_name] = 0
                else:
                    joint_rotation[mapped_name] = item["jointPosition"]

        return robot_position, joint_rotation

    def get_room_dimensions(self, path):
        scene_objects_path = os.path.join(path, "scene_objects.json")
        if os.path.exists(scene_objects_path):
            with open(scene_objects_path, 'r') as f:
                metadata = json.load(f)
            for item in metadata.get("objects", []):
                if item.get("objectId") == "room_interior_bounds":
                    size = item.get("boundsSize", {})
                    return (
                        float(size.get("x", default_room_dimensions[0])),
                        float(size.get("y", default_room_dimensions[1])),
                        float(size.get("z", default_room_dimensions[2])),
                    )

        voxel_metadata_path = os.path.join(path, "voxel_metadata.json")
        if os.path.exists(voxel_metadata_path):
            with open(voxel_metadata_path, 'r') as f:
                metadata = json.load(f)
            voxel_size = metadata.get("voxelSize", {})
            return (
                int(metadata.get("sizeX", 0)) * float(voxel_size.get("x", default_voxel_size)),
                int(metadata.get("sizeY", 0)) * float(voxel_size.get("y", default_voxel_size)),
                int(metadata.get("sizeZ", 0)) * float(voxel_size.get("z", default_voxel_size)),
            )

        return default_room_dimensions

    def get_scene_objects(self, path):
        try:
            metadata = azurion_dataset.load_scene_objects(path)
        except FileNotFoundError:
            return []
        return metadata.get("objects", [])

    def _scene_object_bounds(self, item):
        position = item.get("position", {})
        size = item.get("boundsSize", item.get("scale", {}))
        return {
            "object_id": item.get("objectId", "scene_object"),
            "category": item.get("category", ""),
            "center": np.array([
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            ], dtype=np.float64),
            "size": np.array([
                float(size.get("x", 0.0)),
                float(size.get("y", 0.0)),
                float(size.get("z", 0.0)),
            ], dtype=np.float64),
        }

    def get_table_bounds(self, path):
        table_bounds = []
        for item in self.get_scene_objects(path):
            if item.get("category") != "table":
                continue
            table_bounds.append(self._scene_object_bounds(item))
        return table_bounds

    def get_table_protection_bounds(self, path):
        protected_bounds = []
        for item in self.get_scene_objects(path):
            if item.get("category") in {"table", "room"}:
                continue
            protected_bounds.append(self._scene_object_bounds(item))
        return protected_bounds

class RoomVoxelGrid():
    def __init__(self, resolution=default_voxel_size,width=default_room_dimensions[0], length=default_room_dimensions[1], height=default_room_dimensions[2]):
        self.resolution = resolution
        self.voxel_grid = o3d.geometry.VoxelGrid()
        self.width = width
        self.length = length
        self.height = height
        self.voxel_grid_numpy = None


    def add_point_cloud(self, pcd):
        """Adds a point cloud to the voxel grid."""
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.resolution)

    def clear(self):
        self.voxel_grid.clear()

    def visualize(self,show_walls=True):
        if self.voxel_grid.is_empty():
            print("Error: Voxel grid is empty. Add some point clouds before visualizing.")
            return
        # Use the modern web/PBR viewer
        o3d.visualization.draw([self.voxel_grid], title="Room Visualization (Voxel Grid)", show_ui=True)

    def from_numpy(self, voxel_array):
        origin = np.array([-self.width / 2, 0.0, -self.height / 2])
        occupied = np.argwhere(voxel_array)
        centers = occupied * self.resolution + origin + self.resolution * 0.5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.resolution)

    def get_voxel_grid(self):
        return self.voxel_grid.get_voxels()


    def get_info(self):
        return self.resolution

    def convert_to_numpy(self,debug=False):
        """Converts the voxel grid to a dense numpy array representation."""
        voxels = self.voxel_grid.get_voxels()
        voxel_array= np.zeros((
            grid_count_for_length(self.width, self.resolution),
            grid_count_for_length(self.length, self.resolution),
            grid_count_for_length(self.height, self.resolution)
        ), dtype=bool)
        if debug:
            print("dimensions of the voxel array:", voxel_array.shape);
            
        our_origin = np.array([-self.width / 2, 0.0, -self.height / 2])
        o3d_origin = np.array(self.voxel_grid.origin)
        offset = np.round((o3d_origin - our_origin) / self.resolution).astype(int)

        for voxel in voxels:
            idx = voxel.grid_index + offset
            if all(0 <= idx[i] < voxel_array.shape[i] for i in range(3)):
                voxel_array[idx[0], idx[1], idx[2]] = True
        self.voxel_grid_numpy = voxel_array
        return voxel_array

#-------------------------------Functions------------------------------------
def compute_occlusion_grid(depth_image_paths, intrinsics, extrinsics, resolution=default_voxel_size, room_dimensions=default_room_dimensions):
    grid_shape = (grid_count_for_length(room_dimensions[0], resolution),
                  grid_count_for_length(room_dimensions[1], resolution),
                  grid_count_for_length(room_dimensions[2], resolution))
    origin = np.array([-room_dimensions[0] / 2, 0.0, -room_dimensions[2] / 2])

    xi, yi, zi = np.arange(grid_shape[0]), np.arange(grid_shape[1]), np.arange(grid_shape[2])
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
    voxel_centers_h = np.stack([
        gx.ravel() * resolution + origin[0] + resolution * 0.5,
        gy.ravel() * resolution + origin[1] + resolution * 0.5,
        gz.ravel() * resolution + origin[2] + resolution * 0.5,
        np.ones(grid_shape[0] * grid_shape[1] * grid_shape[2])
    ], axis=0)

    free_flat = np.zeros(voxel_centers_h.shape[1], dtype=bool)
    tolerance = resolution * np.sqrt(3) / 2
    for img_path, intrinsic, extrinsic in zip(depth_image_paths, intrinsics, extrinsics):
        ext = np.array(extrinsic, dtype=np.float64)
        ext[1, :] *= -1
        ext[2, :] *= -1
        ext[:, 2] *= -1

        w, h = intrinsic.width, intrinsic.height
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()

        depth = np.fromfile(img_path, dtype=np.float32).reshape((h, w))
        depth = np.flipud(depth)
        near_clip, far_clip = azurion_dataset.depth_clip_for_raw_path(img_path)

        cam_coords = ext @ voxel_centers_h
        z_cam = cam_coords[2]
        z_safe = np.where(z_cam > 0, z_cam, 1.0)

        ui = np.round(cam_coords[0] / z_safe * fx + cx).astype(int)
        vi = np.round(cam_coords[1] / z_safe * fy + cy).astype(int)

        in_frustum = (z_cam >= near_clip) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        surf_depth = np.where(in_frustum, depth[np.clip(vi, 0, h - 1), np.clip(ui, 0, w - 1)], 0.0)
        surf_valid = np.isfinite(surf_depth) & (surf_depth > near_clip)
        
        is_background = (surf_depth >= far_clip - azurion_dataset.FAR_CLIP_EPSILON)
        effective_depth = np.where(is_background, np.inf, surf_depth)
        free_flat |= in_frustum & surf_valid & (z_cam < effective_depth - tolerance)

    return (~free_flat).reshape(grid_shape)

def compute_blind_spot_grid(depth_image_paths, intrinsics, extrinsics, resolution=default_voxel_size, room_dimensions=default_room_dimensions):
    grid_shape = (grid_count_for_length(room_dimensions[0], resolution),
                  grid_count_for_length(room_dimensions[1], resolution),
                  grid_count_for_length(room_dimensions[2], resolution))
    origin = np.array([-room_dimensions[0] / 2, 0.0, -room_dimensions[2] / 2])

    xi, yi, zi = np.arange(grid_shape[0]), np.arange(grid_shape[1]), np.arange(grid_shape[2])
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
    voxel_centers_h = np.stack([
        gx.ravel() * resolution + origin[0] + resolution * 0.5,
        gy.ravel() * resolution + origin[1] + resolution * 0.5,
        gz.ravel() * resolution + origin[2] + resolution * 0.5,
        np.ones(grid_shape[0] * grid_shape[1] * grid_shape[2])
    ], axis=0)

    seen_flat = np.zeros(voxel_centers_h.shape[1], dtype=bool)

    for img_path, intrinsic, extrinsic in zip(depth_image_paths, intrinsics, extrinsics):
        ext = np.array(extrinsic, dtype=np.float64)
        ext[1, :] *= -1
        ext[2, :] *= -1
        ext[:, 2] *= -1

        w, h = intrinsic.width, intrinsic.height
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()

        near_clip, far_clip = azurion_dataset.depth_clip_for_raw_path(img_path)

        cam_coords = ext @ voxel_centers_h
        z_cam = cam_coords[2]
        z_safe = np.where(z_cam > 0, z_cam, 1.0)

        ui = np.round(cam_coords[0] / z_safe * fx + cx).astype(int)
        vi = np.round(cam_coords[1] / z_safe * fy + cy).astype(int)

        # A voxel is within the camera's FOV if it projects inside the image bounds and is within the near and far clip range
        in_frustum = (z_cam >= near_clip) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z_cam < far_clip)
        seen_flat |= in_frustum

    return (~seen_flat).reshape(grid_shape)

def draft_compute_robot_voxel_mask(robot_model, resolution, room_dimensions, extra_margin_m=0.0):
    touch_margin = resolution * np.sqrt(3) / 2 + extra_margin_m

    grid_shape = (
        grid_count_for_length(room_dimensions[0], resolution),
        grid_count_for_length(room_dimensions[1], resolution),
        grid_count_for_length(room_dimensions[2], resolution),
    )
    origin = np.array([-room_dimensions[0] / 2, 0.0, -room_dimensions[2] / 2])

    xi, yi, zi = np.arange(grid_shape[0]), np.arange(grid_shape[1]), np.arange(grid_shape[2])
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
    centers = np.stack([
        gx.ravel() * resolution + origin[0] + resolution * 0.5,
        gy.ravel() * resolution + origin[1] + resolution * 0.5,
        gz.ravel() * resolution + origin[2] + resolution * 0.5,
    ], axis=1).astype(np.float32)

    query = o3d.core.Tensor(centers, dtype=o3d.core.Dtype.Float32)
    mask_flat = np.zeros(centers.shape[0], dtype=bool)

    for mesh in robot_model.parts.values():
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        occupancy  = scene.compute_occupancy(query).numpy().astype(bool)
        abs_dist   = np.abs(scene.compute_signed_distance(query).numpy())
        mask_flat |= occupancy | (abs_dist <= touch_margin)

    return mask_flat.reshape(grid_shape)

def convert_pointcloud_to_voxelgrid(point_cloud: RoomPointCloud, resolution=default_voxel_size,include_walls=True, room_dimensions=default_room_dimensions):
    pcd_temp = point_cloud
    if not include_walls:
        pcd_temp.crop_point_cloud(x_bound=(-room_dimensions[0]/2, room_dimensions[0]/2),
                                  y_bound=(0, room_dimensions[1]),
                                  z_bound=(-room_dimensions[2]/2, room_dimensions[2]/2),
                                  margin=-default_wall_thickness)
    pcd_temp.color_point_cloud([0.5, 0.5, 0.5])
    voxel_grid = RoomVoxelGrid(resolution=resolution, width=room_dimensions[0], length=room_dimensions[1], height=room_dimensions[2])
    voxel_grid.add_point_cloud(pcd_temp.point_cloud)

    return voxel_grid

def unity_to_o3d_transform(unity_pos, unity_rot_deg=[-90, 90, 0]):
    # 1. Convert Unity Euler (ZXY) to Radian
    rx, ry, rz = np.radians(unity_rot_deg)

    # 2. Build individual matrices (Intrinsic)
    # We negate the Y and Z angles because of the handedness switch
    Rx = o3d.geometry.get_rotation_matrix_from_axis_angle([rx, 0, 0])
    Ry = o3d.geometry.get_rotation_matrix_from_axis_angle([0, -ry, 0])
    Rz = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, -rz])

    # Unity's ZXY sequence: R = Ry @ Rx @ Rz
    R_final = Ry @ Rx @ Rz

    t_final = np.array([unity_pos[0], unity_pos[1], unity_pos[2]])

    T = np.eye(4)
    T[:3, :3] = R_final
    T[:3, 3] = t_final
    return T


#-------------------------------draft decomposition smoke test------------------------------------
def draft_neighbor_offsets(connectivity):
    # this keeps the first split boring and explicit: nearby occupied voxels become one component.
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan == 1:
                    offsets.append((dx, dy, dz))
                elif connectivity == 18 and manhattan <= 2:
                    offsets.append((dx, dy, dz))
                elif connectivity == 26:
                    offsets.append((dx, dy, dz))
    if not offsets:
        raise ValueError("connectivity must be 6, 18, or 26")
    return offsets


def draft_find_connected_components(grid, min_voxels, connectivity):
    # this is the first rough split, not object recognition yet.
    occupied = set(map(tuple, np.argwhere(grid)))
    offsets = draft_neighbor_offsets(connectivity)
    components = []

    while occupied:
        seed = occupied.pop()
        queue = deque([seed])
        component = [seed]

        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in offsets:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in occupied:
                    occupied.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)

        if len(component) >= min_voxels:
            components.append(np.asarray(component, dtype=np.int32))

    components.sort(key=len, reverse=True)
    return components


def draft_component_to_points(component_indices, resolution, room_dimensions):
    # use the same origin convention as RoomVoxelGrid.from_numpy so we do not invent a second frame.
    origin = np.array([-room_dimensions[0] / 2, 0.0, -room_dimensions[2] / 2], dtype=np.float64)
    return component_indices.astype(np.float64) * resolution + origin + resolution * 0.5


def draft_component_color(index):
    # deterministic bright-ish colors, so repeated runs show the same component palette.
    hue = (index * 0.618033988749895) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.72, 0.95), dtype=np.float64)


def draft_colorize_components(component_points, component_indices=None, occlusion_mask=None, color_override=None):
    geometries = []
    for index, points in enumerate(component_points[:MAX_COMPONENTS_TO_DRAW]):
        color = draft_component_color(index) if color_override is None else np.asarray(color_override, dtype=np.float64)
        colors = np.tile(color, (len(points), 1))
        if component_indices is not None and occlusion_mask is not None:
            component = component_indices[index]
            occlusion_flags = occlusion_mask[
                component[:, 0],
                component[:, 1],
                component[:, 2],
            ]
            colors[occlusion_flags] = OCCLUSION_COLOR
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if SHOW_COMPONENT_VOXELS:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=default_voxel_size)
            geometries.append(voxel_grid)
        else:
            geometries.append(pcd)
    return geometries


def draft_component_voxel_corner_points(component_indices, resolution, room_dimensions, margin_m):
    # use voxel corners so the visual hull encloses the same 5 cm cells that MuJoCo receives.
    centers = draft_component_to_points(component_indices, resolution, room_dimensions)
    half_width = resolution * 0.5 + float(margin_m)
    corner_offsets = np.asarray([
        [sx, sy, sz]
        for sx in (-half_width, half_width)
        for sy in (-half_width, half_width)
        for sz in (-half_width, half_width)
    ], dtype=np.float64)
    return (centers[:, None, :] + corner_offsets[None, :, :]).reshape((-1, 3))


def draft_build_component_convex_hulls(
    component_indices,
    resolution,
    room_dimensions,
    kind="entity",
    margin_m=CONVEX_HULL_MARGIN_M,
    color_override=None,
):
    hull_meshes = []
    hull_records = []
    hull_chunks = []
    for object_index, component in enumerate(component_indices[:MAX_COMPONENTS_TO_DRAW]):
        chunks = draft_split_component_for_boxes(
            component,
            resolution,
            room_dimensions,
        )
        color = draft_component_color(object_index) if color_override is None else np.asarray(color_override, dtype=np.float64)
        for chunk_index, chunk in enumerate(chunks):
            points = draft_component_voxel_corner_points(
                chunk,
                resolution,
                room_dimensions,
                margin_m,
            )
            points = np.unique(np.round(points, decimals=6), axis=0)
            if len(points) < 4:
                continue
            try:
                hull = ConvexHull(points)
            except QhullError:
                hull = ConvexHull(points, qhull_options="QJ")

            hull_vertex_indices = np.asarray(hull.vertices, dtype=np.int32)
            index_map = {int(old_index): new_index for new_index, old_index in enumerate(hull_vertex_indices)}
            faces = []
            for simplex in np.asarray(hull.simplices, dtype=np.int32):
                if all(int(index) in index_map for index in simplex):
                    faces.append([index_map[int(index)] for index in simplex])

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points[hull_vertex_indices])
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(color)
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            wireframe.paint_uniform_color(color)
            hull_meshes.append(wireframe)
            hull_records.append({
                "kind": kind,
                "object_index": object_index,
                "chunk_index": chunk_index,
                "voxel_count": len(chunk),
                "vertex_count": len(hull_vertex_indices),
                "face_count": len(faces),
                "margin_m": margin_m,
            })
            hull_chunks.append(chunk)
    return hull_meshes, hull_records, hull_chunks


def draft_log_convex_hulls(hull_records):
    for record in hull_records:
        print(
            f"  {record.get('kind', 'entity')} {record['object_index']:02d} hull chunk {record.get('chunk_index', 0):02d}: "
            f"voxels={record['voxel_count']} "
            f"vertices={record['vertex_count']} "
            f"faces={record['face_count']} "
            f"margin={record['margin_m']:.2f}m"
        )


def draft_fit_oriented_box(points, resolution, margin_m):
    # fit yaw from x/z only and keep the box upright, so floor objects never get pitch/roll tilt.
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("cannot fit a box around zero points")

    centroid = points.mean(axis=0)
    centered = points - centroid

    if len(points) >= 3:
        horizontal = centered[:, [0, 2]]
        covariance = np.cov(horizontal, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        horizontal_axis = eigenvectors[:, order[0]]
        x_axis = np.array([horizontal_axis[0], 0.0, horizontal_axis[1]], dtype=np.float64)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            x_axis /= x_norm
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis /= z_norm
    rotation = np.column_stack([x_axis, y_axis, z_axis])

    # keep the frame right-handed so open3d and later mujoco export get a proper rotation.
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0

    local_points = centered @ rotation
    min_local = local_points.min(axis=0)
    max_local = local_points.max(axis=0)
    local_center = (min_local + max_local) * 0.5
    unpadded_extent = np.maximum(max_local - min_local, resolution)
    padded_extent = unpadded_extent + (2.0 * margin_m)
    world_center = centroid + local_center @ rotation.T

    box = o3d.geometry.OrientedBoundingBox(world_center, rotation, padded_extent)
    return box, unpadded_extent, padded_extent, rotation, local_points


def draft_split_component_for_boxes(component, resolution, room_dimensions, depth=0, max_boxes=None):
    # this optional split is a simple upright kd-tree split, not semantic recognition.
    if not SPLIT_OBJECTS_INTO_SMALLER_BOXES:
        return [component]

    max_boxes = SPLIT_MAX_BOXES_PER_OBJECT if max_boxes is None else int(max_boxes)
    chunks = [component]
    split_depths = [depth]

    while len(chunks) < max_boxes:
        best_index = None
        best_edge = 0.0
        best_data = None

        for index, chunk in enumerate(chunks):
            if split_depths[index] >= SPLIT_MAX_RECURSION_DEPTH:
                continue
            if len(chunk) < (2 * SPLIT_MIN_VOXELS_PER_BOX):
                continue

            points = draft_component_to_points(chunk, resolution, room_dimensions)
            _, unpadded_extent, _, _, local_points = draft_fit_oriented_box(
                points,
                resolution,
                margin_m=0.0,
            )
            longest_axis = int(np.argmax(unpadded_extent))
            longest_edge = float(unpadded_extent[longest_axis])
            if longest_edge > best_edge:
                best_index = index
                best_edge = longest_edge
                best_data = (local_points, longest_axis)

        if best_index is None or best_edge <= SPLIT_MAX_BOX_EDGE_M:
            break

        local_points, longest_axis = best_data
        split_values = local_points[:, longest_axis]
        split_at = float(np.median(split_values))
        left_mask = split_values <= split_at
        right_mask = ~left_mask

        if np.count_nonzero(left_mask) < SPLIT_MIN_VOXELS_PER_BOX:
            break
        if np.count_nonzero(right_mask) < SPLIT_MIN_VOXELS_PER_BOX:
            break

        chunk_to_split = chunks.pop(best_index)
        chunk_depth = split_depths.pop(best_index)
        chunks.append(chunk_to_split[left_mask])
        chunks.append(chunk_to_split[right_mask])
        split_depths.append(chunk_depth + 1)
        split_depths.append(chunk_depth + 1)

    return chunks


def draft_build_component_bounding_boxes(
    component_indices,
    resolution,
    room_dimensions,
    kind="object",
    margin_m=OBJECT_BOX_MARGIN_M,
    max_boxes_per_component=None,
    color_override=None,
    occlusion_mask=None,
):
    boxes = []
    box_records = []
    for object_index, component in enumerate(component_indices[:MAX_COMPONENTS_TO_DRAW]):
        chunks = draft_split_component_for_boxes(
            component,
            resolution,
            room_dimensions,
            max_boxes=max_boxes_per_component,
        )
        color = draft_component_color(object_index) if color_override is None else np.asarray(color_override, dtype=np.float64)
        for box_index, chunk in enumerate(chunks):
            points = draft_component_to_points(chunk, resolution, room_dimensions)
            box, unpadded_extent, padded_extent, _, _ = draft_fit_oriented_box(
                points,
                resolution,
                margin_m=margin_m,
            )
            tilt_degrees = draft_box_tilt_degrees(box)
            surface_hint = draft_surface_hint(points, room_dimensions)
            occlusion_voxel_count = 0
            if occlusion_mask is not None:
                occlusion_voxel_count = int(np.count_nonzero(occlusion_mask[
                    chunk[:, 0],
                    chunk[:, 1],
                    chunk[:, 2],
                ]))
            box.color = color
            boxes.append(box)
            box_records.append({
                "kind": kind,
                "object_index": object_index,
                "box_index": box_index,
                "voxel_count": len(chunk),
                "occlusion_voxel_count": occlusion_voxel_count,
                "center": np.asarray(box.center),
                "unpadded_extent": unpadded_extent,
                "padded_extent": padded_extent,
                "tilt_degrees": tilt_degrees,
                "surface_hint": surface_hint,
            })
    return boxes, box_records


def draft_box_tilt_degrees(box):
    # 0 degrees means one box axis is aligned with world up. larger means the proxy is leaning.
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_alignment = np.abs(np.asarray(box.R, dtype=np.float64).T @ up)
    best_alignment = float(np.clip(np.max(axis_alignment), 0.0, 1.0))
    return float(np.degrees(np.arccos(best_alignment)))


def draft_surface_hint(points, room_dimensions):
    if len(points) == 0:
        return "unknown"
    y_values = np.asarray(points, dtype=np.float64)[:, 1]
    if float(np.min(y_values)) <= FLOOR_REMOVAL_HEIGHT_M + default_voxel_size:
        return "floor"
    if float(np.max(y_values)) >= room_dimensions[1] - CEILING_REMOVAL_MARGIN_M - default_voxel_size:
        return "ceiling"
    return "floating"


def draft_remove_room_boundary_voxels(grid, resolution, room_dimensions):
    # this strips the giant room shell so the component split focuses on table/props/leftover obstacles.
    cleaned = np.array(grid, copy=True)

    size_x, size_y, size_z = cleaned.shape
    origin = np.array([-room_dimensions[0] / 2, 0.0, -room_dimensions[2] / 2], dtype=np.float64)
    x_centers = np.arange(size_x, dtype=np.float64) * resolution + origin[0] + resolution * 0.5
    y_centers = np.arange(size_y, dtype=np.float64) * resolution + origin[1] + resolution * 0.5
    z_centers = np.arange(size_z, dtype=np.float64) * resolution + origin[2] + resolution * 0.5

    x_half = room_dimensions[0] / 2.0
    z_half = room_dimensions[2] / 2.0
    boundary_x = np.abs(x_centers) >= (x_half - ROOM_BOUNDARY_MARGIN_M)
    boundary_z = np.abs(z_centers) >= (z_half - ROOM_BOUNDARY_MARGIN_M)
    floor_y = y_centers <= FLOOR_REMOVAL_HEIGHT_M
    ceiling_y = y_centers >= (room_dimensions[1] - CEILING_REMOVAL_MARGIN_M)

    cleaned[boundary_x, :, :] = False
    cleaned[:, :, boundary_z] = False
    cleaned[:, floor_y, :] = False
    cleaned[:, ceiling_y, :] = False
    return cleaned


def draft_bounds_mask(grid_shape, bounds, resolution, room_dimensions, margin_m):
    mask = np.zeros(grid_shape, dtype=bool)
    if not bounds:
        return mask

    size_x, size_y, size_z = grid_shape
    origin = np.array([-room_dimensions[0] / 2, 0.0, -room_dimensions[2] / 2], dtype=np.float64)
    x_centers = np.arange(size_x, dtype=np.float64) * resolution + origin[0] + resolution * 0.5
    y_centers = np.arange(size_y, dtype=np.float64) * resolution + origin[1] + resolution * 0.5
    z_centers = np.arange(size_z, dtype=np.float64) * resolution + origin[2] + resolution * 0.5

    for item in bounds:
        size = np.asarray(item["size"], dtype=np.float64)
        if np.any(size <= 0.0):
            continue
        half_size = size * 0.5 + margin_m
        center = item["center"]
        x_mask = (x_centers >= center[0] - half_size[0]) & (x_centers <= center[0] + half_size[0])
        y_mask = (y_centers >= center[1] - half_size[1]) & (y_centers <= center[1] + half_size[1])
        z_mask = (z_centers >= center[2] - half_size[2]) & (z_centers <= center[2] + half_size[2])
        mask[np.ix_(x_mask, y_mask, z_mask)] = True

    return mask


def draft_table_low_support_mask(grid_shape, table_bounds, resolution, room_dimensions):
    if not table_bounds:
        return np.zeros(grid_shape, dtype=bool)

    top_parts = [
        item for item in table_bounds
        if item["object_id"] in {"table_top", "rail_left", "rail_right"}
    ]
    footprint_parts = top_parts or table_bounds

    min_x = min(float(item["center"][0] - item["size"][0] * 0.5) for item in footprint_parts) - TABLE_LOW_SUPPORT_MARGIN_M
    max_x = max(float(item["center"][0] + item["size"][0] * 0.5) for item in footprint_parts) + TABLE_LOW_SUPPORT_MARGIN_M
    min_z = min(float(item["center"][2] - item["size"][2] * 0.5) for item in footprint_parts) - TABLE_LOW_SUPPORT_MARGIN_M
    max_z = max(float(item["center"][2] + item["size"][2] * 0.5) for item in footprint_parts) + TABLE_LOW_SUPPORT_MARGIN_M

    size_x, size_y, size_z = grid_shape
    origin = np.array([-room_dimensions[0] / 2, 0.0, -room_dimensions[2] / 2], dtype=np.float64)
    x_centers = np.arange(size_x, dtype=np.float64) * resolution + origin[0] + resolution * 0.5
    y_centers = np.arange(size_y, dtype=np.float64) * resolution + origin[1] + resolution * 0.5
    z_centers = np.arange(size_z, dtype=np.float64) * resolution + origin[2] + resolution * 0.5

    x_mask = (x_centers >= min_x) & (x_centers <= max_x)
    y_mask = y_centers <= TABLE_LOW_SUPPORT_HEIGHT_M
    z_mask = (z_centers >= min_z) & (z_centers <= max_z)

    mask = np.zeros(grid_shape, dtype=bool)
    mask[np.ix_(x_mask, y_mask, z_mask)] = True
    return mask


def draft_remove_table_voxels(grid, table_bounds, resolution, room_dimensions, protected_bounds=None, fixture_mask=None):
    # Unity table entries are exported from actual rendered table bounds; the optional
    # fixture mask comes from Unity's scene/props/robot voxel split and catches fixture surfaces.
    if not table_bounds and fixture_mask is None:
        return np.array(grid, copy=True), 0

    cleaned = np.array(grid, copy=True)
    before = int(np.count_nonzero(cleaned))
    table_mask = draft_bounds_mask(
        cleaned.shape,
        table_bounds,
        resolution,
        room_dimensions,
        TABLE_REMOVAL_MARGIN_M,
    )
    if fixture_mask is not None and fixture_mask.shape == cleaned.shape:
        table_mask = np.logical_or(table_mask, fixture_mask)
    cleaned[table_mask] = False
    if table_bounds and REMOVE_TABLE_LOW_SUPPORT_CLEANUP:
        cleaned[draft_table_low_support_mask(cleaned.shape, table_bounds, resolution, room_dimensions)] = False

    if table_bounds and REMOVE_TABLE_EDGE_COMPONENTS:
        cleaned = draft_remove_small_table_edge_components(
            cleaned,
            table_bounds,
            resolution,
            room_dimensions,
            protected_bounds=protected_bounds,
        )

    removed = before - int(np.count_nonzero(cleaned))
    return cleaned, removed


def draft_load_unity_table_fixture_mask(sample_path, grid_shape):
    try:
        scene_metadata, scene_raw = azurion_dataset.load_unity_voxels(sample_path, "scene")
        props_metadata, props_raw = azurion_dataset.load_unity_voxels(sample_path, "props")
        robot_metadata, robot_raw = azurion_dataset.load_unity_voxels(sample_path, "robot")
    except (FileNotFoundError, ValueError, KeyError):
        return None

    scene_grid = azurion_dataset.reshape_unity_voxel_grid(scene_metadata, scene_raw)
    props_grid = azurion_dataset.reshape_unity_voxel_grid(props_metadata, props_raw)
    robot_grid = azurion_dataset.reshape_unity_voxel_grid(robot_metadata, robot_raw)
    if scene_grid.shape != grid_shape or props_grid.shape != grid_shape or robot_grid.shape != grid_shape:
        return None

    fixture_mask = np.logical_and(scene_grid, ~np.logical_or(props_grid, robot_grid))
    if TABLE_FIXTURE_EXPAND_VOXELS > 0:
        fixture_mask = draft_expand_grid(
            fixture_mask,
            TABLE_FIXTURE_EXPAND_VOXELS,
            COMPONENT_CONNECTIVITY,
        )

    protected_props = props_grid
    if TABLE_FIXTURE_PROTECT_PROPS_EXPAND_VOXELS > 0:
        protected_props = draft_expand_grid(
            protected_props,
            TABLE_FIXTURE_PROTECT_PROPS_EXPAND_VOXELS,
            COMPONENT_CONNECTIVITY,
        )
    return np.logical_and(fixture_mask, ~protected_props)


def draft_remove_small_table_edge_components(grid, table_bounds, resolution, room_dimensions, protected_bounds=None):
    # after the main cut, tiny low fragments just outside the table mask are almost always table edges.
    if not table_bounds:
        return np.array(grid, copy=True)

    cleaned = np.array(grid, copy=True)
    edge_mask = draft_bounds_mask(
        cleaned.shape,
        table_bounds,
        resolution,
        room_dimensions,
        TABLE_EDGE_COMPONENT_MARGIN_M,
    )
    protected_mask = draft_bounds_mask(
        cleaned.shape,
        protected_bounds or [],
        resolution,
        room_dimensions,
        0.0,
    )
    components = draft_find_connected_components(
        cleaned,
        min_voxels=1,
        connectivity=COMPONENT_CONNECTIVITY,
    )

    for component in components:
        if len(component) > TABLE_EDGE_COMPONENT_MAX_VOXELS:
            continue
        touches_edge_mask = np.any(edge_mask[
            component[:, 0],
            component[:, 1],
            component[:, 2],
        ])
        if not touches_edge_mask:
            continue
        if np.any(protected_mask[
            component[:, 0],
            component[:, 1],
            component[:, 2],
        ]):
            continue
        cleaned[component[:, 0], component[:, 1], component[:, 2]] = False

    return cleaned


def draft_fill_one_voxel_gaps(grid, iterations):
    # this bridges tiny single-voxel cracks, while keeping the original resolution intact.
    filled = np.array(grid, copy=True)
    total_added = 0

    for _ in range(max(0, int(iterations))):
        before = int(np.count_nonzero(filled))
        gap_candidates = np.zeros_like(filled, dtype=bool)

        minus_x = np.zeros_like(filled, dtype=bool)
        plus_x = np.zeros_like(filled, dtype=bool)
        minus_x[1:, :, :] = filled[:-1, :, :]
        plus_x[:-1, :, :] = filled[1:, :, :]
        gap_candidates |= (~filled) & minus_x & plus_x

        minus_y = np.zeros_like(filled, dtype=bool)
        plus_y = np.zeros_like(filled, dtype=bool)
        minus_y[:, 1:, :] = filled[:, :-1, :]
        plus_y[:, :-1, :] = filled[:, 1:, :]
        gap_candidates |= (~filled) & minus_y & plus_y

        minus_z = np.zeros_like(filled, dtype=bool)
        plus_z = np.zeros_like(filled, dtype=bool)
        minus_z[:, :, 1:] = filled[:, :, :-1]
        plus_z[:, :, :-1] = filled[:, :, 1:]
        gap_candidates |= (~filled) & minus_z & plus_z

        filled |= gap_candidates
        added = int(np.count_nonzero(filled)) - before
        total_added += added
        if added == 0:
            break

    return filled, total_added


def draft_expand_grid(grid, radius_voxels, connectivity):
    expanded = np.array(grid, copy=True)
    offsets = draft_neighbor_offsets(connectivity)
    for _ in range(max(0, int(radius_voxels))):
        next_grid = np.array(expanded, copy=True)
        occupied = np.argwhere(expanded)
        for dx, dy, dz in offsets:
            shifted = occupied + np.array([dx, dy, dz], dtype=np.int64)
            valid = (
                (shifted[:, 0] >= 0) & (shifted[:, 0] < expanded.shape[0]) &
                (shifted[:, 1] >= 0) & (shifted[:, 1] < expanded.shape[1]) &
                (shifted[:, 2] >= 0) & (shifted[:, 2] < expanded.shape[2])
            )
            shifted = shifted[valid]
            next_grid[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = True
        expanded = next_grid
    return expanded


def draft_find_touching_occlusion_components(visible_grid, occlusion_grid, min_voxels, connectivity):
    # occlusion_grid means "not proven free"; this legacy helper keeps only pockets attached to visible obstacles.
    attached_components, stats = draft_find_occlusion_zone_components(
        visible_grid,
        occlusion_grid,
        min_voxels,
        connectivity,
        include_detached=False,
    )
    return attached_components, stats


def draft_find_occlusion_zone_components(visible_grid, occlusion_grid, min_voxels, connectivity, include_detached):
    # this is the colleague-style not-free grid, corrected for the new intrinsics and room dimensions.
    # we remove the directly visible surface first, then keep shadow pockets. attached pockets inflate objects;
    # detached pockets become their own conservative no-go entities.
    visible_clearance = draft_expand_grid(
        visible_grid,
        OCCLUSION_SURFACE_CLEARANCE_VOXELS,
        connectivity,
    )
    touch_radius = (
        OCCLUSION_SURFACE_CLEARANCE_VOXELS
        + OCCLUSION_ATTACHMENT_RADIUS_VOXELS
    )
    expanded_visible = draft_expand_grid(
        visible_grid,
        touch_radius,
        connectivity,
    )
    candidate_grid = np.logical_and(occlusion_grid, ~visible_clearance)
    candidate_components = draft_find_connected_components(
        candidate_grid,
        min_voxels=min_voxels,
        connectivity=connectivity,
    )

    attached_components = []
    detached_components = []
    skipped_large_components = 0

    for component in candidate_components:
        if len(component) > OCCLUSION_MAX_COMPONENT_VOXELS:
            skipped_large_components += 1
            continue
        touches_visible = np.any(expanded_visible[
            component[:, 0],
            component[:, 1],
            component[:, 2],
        ])
        if touches_visible:
            attached_components.append(component)
            continue
        if not include_detached:
            continue
        detached_components.append(component)

    selected_components = attached_components + detached_components
    return selected_components, {
        "candidate_components": len(candidate_components),
        "selected_components": len(selected_components),
        "attached_components": len(attached_components),
        "detached_components": len(detached_components),
        "attached_voxels": int(sum(len(component) for component in attached_components)),
        "detached_voxels": int(sum(len(component) for component in detached_components)),
        "selected_voxels": int(sum(len(component) for component in selected_components)),
        "skipped_large_components": skipped_large_components,
    }


def draft_occlusion_components_to_mask(components, grid_shape):
    mask = np.zeros(grid_shape, dtype=bool)
    for component in components:
        mask[component[:, 0], component[:, 1], component[:, 2]] = True
    return mask


def draft_build_occlusion_entity_mask(visible_grid, occlusion_components, connectivity):
    # selected shadows are brown, but they still join the entity grid so one object+shadow can get one box.
    selected_mask = draft_occlusion_components_to_mask(occlusion_components, visible_grid.shape)
    if not np.any(selected_mask):
        return selected_mask

    # the surface-clearance step intentionally creates a one-voxel gap around visible surfaces.
    # add only the small bridge that touches both sides, so attached shadows really belong to their object.
    expanded_shadow = draft_expand_grid(
        selected_mask,
        OCCLUSION_SURFACE_CLEARANCE_VOXELS,
        connectivity,
    )
    expanded_visible = draft_expand_grid(
        visible_grid,
        OCCLUSION_SURFACE_CLEARANCE_VOXELS,
        connectivity,
    )
    bridge_mask = np.logical_and(expanded_shadow, expanded_visible)
    return np.logical_or(selected_mask, bridge_mask)


def draft_merge_touching_occlusions(visible_grid, occlusion_grid, min_voxels, connectivity):
    # optional legacy behavior: inflate visible objects with attached unknown pockets.
    attached_components, stats = draft_find_touching_occlusion_components(
        visible_grid,
        occlusion_grid,
        min_voxels,
        connectivity,
    )

    merged_grid = np.array(visible_grid, copy=True)
    merged_voxels = 0
    for component in attached_components:
        already_present = merged_grid[
            component[:, 0],
            component[:, 1],
            component[:, 2],
        ]
        merged_grid[component[:, 0], component[:, 1], component[:, 2]] = True
        merged_voxels += int(len(component) - np.count_nonzero(already_present))

    return merged_grid, {
        "candidate_components": stats["candidate_components"],
        "merged_components": stats["attached_components"],
        "merged_voxels": merged_voxels,
        "skipped_large_components": stats["skipped_large_components"],
    }


def draft_print_component_stats(label, components, occupied_count):
    largest = len(components[0]) if components else 0
    print(f"{label} occupied voxel count: {occupied_count}")
    print(f"{label} surviving components: {len(components)}")
    print(f"{label} largest component voxels: {largest}")


def draft_log_box_proxies(box_records):
    # these oriented boxes are the first mujoco-friendly proxy idea, not final object recognition.
    for record in box_records:
        center_m = record["center"]
        size_m = record["padded_extent"]
        raw_size_m = record["unpadded_extent"]
        tilt_note = ""
        if (
            REPORT_BOX_TILT_DIAGNOSTICS
            and record["surface_hint"] in ("floor", "ceiling")
            and record["tilt_degrees"] > BOX_TILT_NOTE_DEGREES
        ):
            tilt_note = " tilt_note=surface_object_not_level"
        print(
            f"  {record.get('kind', 'object')} {record['object_index']:02d} box {record['box_index']:02d}: "
            f"voxels={record['voxel_count']} "
            f"occlusion_voxels={record.get('occlusion_voxel_count', 0)} "
            f"center=({center_m[0]:.2f}, {center_m[1]:.2f}, {center_m[2]:.2f}) "
            f"raw_size=({raw_size_m[0]:.2f}, {raw_size_m[1]:.2f}, {raw_size_m[2]:.2f}) "
            f"padded_size=({size_m[0]:.2f}, {size_m[1]:.2f}, {size_m[2]:.2f}) "
            f"tilt={record['tilt_degrees']:.1f}deg "
            f"surface={record['surface_hint']}{tilt_note}"
        )


def draft_disable_triangle_mesh_paint_for_local_open3d():
    # this local open3d build segfaults on TriangleMesh.paint_uniform_color for these stl hulls.
    # the copied RobotModel still calls the same line, but this makes the cosmetic paint call a no-op.
    def safe_paint_uniform_color(mesh, color):
        return mesh

    o3d.geometry.TriangleMesh.paint_uniform_color = safe_paint_uniform_color


def draft_require_stable_runtime():
    # apple's /usr/bin/python3.9 plus open3d 0.18 crashed repeatedly on this machine.
    # fail loudly before native open3d work instead of letting macos show "python quit unexpectedly".
    if not REQUIRE_STABLE_OPEN3D_RUNTIME:
        return

    version_text = getattr(o3d, "__version__", "0.0.0")
    major_minor = tuple(int(part) for part in version_text.split(".")[:2])
    if sys.version_info < (3, 11) or major_minor < (0, 19):
        raise RuntimeError(
            "this draft needs the stable local runtime: /opt/homebrew/bin/python3.11 "
            "with open3d>=0.19. vscode is probably using apple python 3.9/open3d 0.18, "
            "which is what caused 'python quit unexpectedly'."
        )


def draft_add_depth_image_safe(point_cloud_holder, image_path, intrinsic, extrinsic=np.eye(4), add_noise=False):
    # the copied add_DepthImage path uses open3d's native create_from_depth_image.
    # on this machine that native call crashes on the new 1920x1080 raws, so this does the same pinhole math in numpy.
    width, height = intrinsic.width, intrinsic.height
    depth_data = np.fromfile(image_path, dtype=np.float32).reshape((height, width))

    near_clip, far_clip = azurion_dataset.depth_clip_for_raw_path(image_path)
    print("Near clip:", near_clip, "Far clip:", far_clip)
    depth_data[~azurion_dataset.valid_depth_mask(depth_data, near_clip, far_clip)] = 0.0

    if add_noise:
        depth_data = point_cloud_holder.apply_noise(depth_data, sigma=0.02, dropout_pixel=0.01)

    depth_data = np.ascontiguousarray(np.flipud(depth_data))

    ext = np.array(extrinsic, dtype=np.float64)
    ext[1, :] *= -1
    ext[2, :] *= -1
    ext[:, 2] *= -1
    camera_to_world = np.linalg.inv(np.ascontiguousarray(ext, dtype=np.float64))

    stride = max(1, int(DRAFT_DEPTH_PIXEL_STRIDE))
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    uu, vv = np.meshgrid(cols, rows)
    sampled_depth = depth_data[::stride, ::stride]

    near_clip, far_clip = azurion_dataset.depth_clip_for_raw_path(image_path)
    valid = azurion_dataset.valid_depth_mask(sampled_depth, near_clip, far_clip)
    if not np.any(valid):
        return

    z = sampled_depth[valid].astype(np.float64)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    camera_points = np.stack([x, y, z, np.ones_like(z)], axis=0)
    world_points = (camera_to_world @ camera_points).T[:, :3]
    world_points = world_points[np.all(np.isfinite(world_points), axis=1)]
    world_points = np.ascontiguousarray(world_points, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points)
    point_cloud_holder.point_cloud += pcd


def draft_crop_point_cloud_safe(point_cloud_holder, x_bound, y_bound, z_bound, margin=0.5):
    # same bounds as the copied crop_point_cloud method, but the masking happens in numpy.
    if point_cloud_holder.point_cloud.is_empty():
        print("Error: Point cloud is empty. Cannot crop.")
        return

    points = np.asarray(point_cloud_holder.point_cloud.points)
    min_bound = np.array([
        x_bound[0] - margin,
        y_bound[0] - margin,
        z_bound[0] - margin,
    ], dtype=np.float64)
    max_bound = np.array([
        x_bound[1] + margin,
        y_bound[1] + margin,
        z_bound[1] + margin,
    ], dtype=np.float64)
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    cropped_points = np.ascontiguousarray(points[mask], dtype=np.float64)

    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
    point_cloud_holder.point_cloud = cropped_pcd


def draft_unity_to_o3d_transform_safe(unity_pos, unity_rot_deg=[-90, 90, 0]):
    # same sequence as unity_to_o3d_transform, just avoiding open3d's crashing rotation helper.
    rx, ry, rz = np.radians(unity_rot_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(-ry), np.sin(-ry)
    cz, sz = np.cos(-rz), np.sin(-rz)

    rx_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ], dtype=np.float64)
    ry_matrix = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=np.float64)
    rz_matrix = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = ry_matrix @ rx_matrix @ rz_matrix
    transform[:3, 3] = np.array([unity_pos[0], unity_pos[1], unity_pos[2]], dtype=np.float64)
    return transform


def draft_process_sample(dataset, sample_index):
    sample_path = dataset.get_samplepath(sample_index)
    if sample_path is None:
        raise RuntimeError(f"no sample folder found at index {sample_index}")

    print("")
    print("=" * 72)
    print("sample index:", sample_index)
    print("selected sample path:", sample_path)
    print("depth pixel stride:", DRAFT_DEPTH_PIXEL_STRIDE)
    print(f"object box margin: {OBJECT_BOX_MARGIN_M:.2f} m")
    print(f"convex hull margin: {CONVEX_HULL_MARGIN_M:.2f} m")
    print("show convex hulls:", SHOW_COMPONENT_CONVEX_HULLS)
    print("split objects into smaller boxes:", SPLIT_OBJECTS_INTO_SMALLER_BOXES)
    print("max boxes per object:", SPLIT_MAX_BOXES_PER_OBJECT)
    print("fill one-voxel gaps:", FILL_ONE_VOXEL_GAPS)
    print("show full voxel grid:", SHOW_FULL_VOXEL_GRID)
    print("show blind spots:", SHOW_BLIND_SPOTS)
    print("remove table in decomposition:", REMOVE_TABLE_IN_DECOMPOSITION)
    print("include table in decomposition:", INCLUDE_TABLE_IN_DECOMPOSITION)
    print("remove small table edge components:", REMOVE_TABLE_EDGE_COMPONENTS)
    print("merge touching occlusions with objects:", MERGE_OCCLUSIONS_WITH_OBJECTS)
    print("include detached occlusion zones:", INCLUDE_DETACHED_OCCLUSION_ZONES)
    print("build separate occlusion zone boxes:", BUILD_OCCLUSION_ZONE_BOXES)
    if SPLIT_OBJECTS_INTO_SMALLER_BOXES:
        print(f"split max box edge: {SPLIT_MAX_BOX_EDGE_M:.2f} m")
    depth_image_paths, intrinsics, extrinsics = dataset.get_data(sample_path)
    robot_position, joint_rotations = dataset.get_robot_pose(sample_path)
    room_dimensions = dataset.get_room_dimensions(sample_path)
    table_bounds = dataset.get_table_bounds(sample_path)
    table_protection_bounds = dataset.get_table_protection_bounds(sample_path)
    print("room dimensions:", room_dimensions)
    print("table metadata parts:", len(table_bounds))
    print("table-protected non-table objects:", len(table_protection_bounds))

    if not depth_image_paths or robot_position is None or joint_rotations is None:
        raise RuntimeError("could not load the depth data or robot pose")

    draft_disable_triangle_mesh_paint_for_local_open3d()
    robot_model = RobotModel(ROBOT_HULLS_FOLDER, ROBOT_URDF_PATH)
    room_point_cloud = RoomPointCloud()

    for img_path, intrinsic, extrinsic in zip(depth_image_paths, intrinsics, extrinsics):
        if DRAFT_USE_SAFE_DEPTH_PROJECTION:
            draft_add_depth_image_safe(room_point_cloud, img_path, intrinsic, extrinsic, add_noise=False)
        else:
            room_point_cloud.add_DepthImage(img_path, intrinsic, extrinsic,add_noise=False)

    draft_crop_point_cloud_safe(room_point_cloud,
                                  x_bound=(-room_dimensions[0]/2, room_dimensions[0]/2),
                                  y_bound=(0, room_dimensions[1] - 0.6),
                                  z_bound=(-room_dimensions[2]/2, room_dimensions[2]/2),
                                  margin=0.5)

    points_before_robot_removal = len(room_point_cloud.point_cloud.points)
    print("point count before robot removal:", points_before_robot_removal)

    transform = draft_unity_to_o3d_transform_safe(robot_position, unity_rot_deg=[-90, -90, 0])
    robot_model.apply_urdf_fk_pose(joint_values=joint_rotations, world_transform=transform)

    # the existing function only removes points when make_red is false.
    room_point_cloud.filter_robot_points(robot_model, make_red=False, extra_margin_m=0.01)
    points_after_robot_removal = len(room_point_cloud.point_cloud.points)
    robot_points_removed = points_before_robot_removal - points_after_robot_removal
    print("robot points removed:", robot_points_removed)
    print("point count after robot removal:", points_after_robot_removal)

    converted_voxel_grid = convert_pointcloud_to_voxelgrid(room_point_cloud, include_walls=True, room_dimensions=room_dimensions)
    occupied_grid = converted_voxel_grid.convert_to_numpy(debug=True)
    raw_occupied_count = int(np.count_nonzero(occupied_grid))
    print("raw occupied grid voxel count:", raw_occupied_count)
    table_fixture_mask = draft_load_unity_table_fixture_mask(sample_path, occupied_grid.shape)
    if table_fixture_mask is not None:
        print("unity table fixture mask voxels:", int(np.count_nonzero(table_fixture_mask)))

    print("computing occlusion diagnostic with the existing function...")
    occlusion_grid = compute_occlusion_grid(depth_image_paths, intrinsics, extrinsics, room_dimensions=room_dimensions)
    
    print("removing robot body from occlusion grid...")
    robot_voxel_mask = draft_compute_robot_voxel_mask(robot_model, default_voxel_size, room_dimensions, extra_margin_m=0.01)
    occlusion_grid &= ~robot_voxel_mask
    
    print("removing blind spots from occlusion grid...")
    raw_blind_spots = compute_blind_spot_grid(depth_image_paths, intrinsics, extrinsics, resolution=default_voxel_size, room_dimensions=room_dimensions)
    occlusion_grid &= ~raw_blind_spots
    
    occlusion_total = int(occlusion_grid.size)
    occlusion_not_free = int(np.count_nonzero(occlusion_grid))
    occlusion_percent = (occlusion_not_free / occlusion_total * 100.0) if occlusion_total else 0.0
    print("occlusion total voxels:", occlusion_total)
    print("occlusion not-free voxels:", occlusion_not_free)
    print(f"occlusion not-free percent: {occlusion_percent:.2f}%")

    blind_spot_boxes = []
    blind_spot_box_records = []
    blind_spot_geometries = []
    blind_spot_hulls = []
    blind_spot_hull_records = []
    blind_spot_hull_chunks = []
    blind_spot_components = []
    if SHOW_BLIND_SPOTS:
        print("processing blind spot geometries...")
        blind_spot_grid = np.array(raw_blind_spots, copy=True)
        
            
        if REMOVE_TABLE_IN_DECOMPOSITION:
            blind_spot_grid, _ = draft_remove_table_voxels(
                blind_spot_grid,
                table_bounds,
                default_voxel_size,
                room_dimensions,
                protected_bounds=table_protection_bounds,
                fixture_mask=table_fixture_mask,
            )
            
        blind_spot_components = draft_find_connected_components(
            blind_spot_grid,
            min_voxels=OCCLUSION_MIN_COMPONENT_VOXELS,
            connectivity=COMPONENT_CONNECTIVITY,
        )
        print("blind spot candidate components:", len(blind_spot_components))
        
        blind_spot_points = [
            draft_component_to_points(component, default_voxel_size, room_dimensions)
            for component in blind_spot_components
        ]
        blind_spot_geometries = draft_colorize_components(
            blind_spot_points,
            color_override=BLIND_SPOT_COLOR,
        )
        
        blind_spot_boxes, blind_spot_box_records = draft_build_component_bounding_boxes(
            blind_spot_components,
            default_voxel_size,
            room_dimensions,
            kind="blind_spot",
            margin_m=OCCLUSION_BOX_MARGIN_M,
            max_boxes_per_component=OCCLUSION_MAX_BOXES_PER_ZONE,
            color_override=BLIND_SPOT_COLOR,
        )
        print("blind spot boxes produced:", len(blind_spot_boxes))

        blind_spot_hulls, blind_spot_hull_records, blind_spot_hull_chunks = draft_build_component_convex_hulls(
            blind_spot_components,
            default_voxel_size,
            room_dimensions,
            kind="blind_spot",
            margin_m=CONVEX_HULL_MARGIN_M,
            color_override=BLIND_SPOT_COLOR,
        )
        print("blind spot convex hulls produced:", len(blind_spot_hulls))

    decomposition_grid = occupied_grid
    clean_occlusion_grid = occlusion_grid
    if REMOVE_ROOM_BOUNDARY_VOXELS:
        decomposition_grid = draft_remove_room_boundary_voxels(
            occupied_grid,
            default_voxel_size,
            room_dimensions,
        )
        stripped_count = raw_occupied_count - int(np.count_nonzero(decomposition_grid))
        print("room boundary voxels stripped:", stripped_count)
        clean_occlusion_grid = draft_remove_room_boundary_voxels(
            clean_occlusion_grid,
            default_voxel_size,
            room_dimensions,
        )

    if FILL_ONE_VOXEL_GAPS:
        decomposition_grid, filled_gap_voxels = draft_fill_one_voxel_gaps(
            decomposition_grid,
            GAP_FILL_ITERATIONS,
        )
        print("one-voxel gaps filled:", filled_gap_voxels)

    if REMOVE_TABLE_IN_DECOMPOSITION:
        decomposition_grid, removed_table_voxels = draft_remove_table_voxels(
            decomposition_grid,
            table_bounds,
            default_voxel_size,
            room_dimensions,
            protected_bounds=table_protection_bounds,
            fixture_mask=table_fixture_mask,
        )
        clean_occlusion_grid, removed_table_occlusion_voxels = draft_remove_table_voxels(
            clean_occlusion_grid,
            table_bounds,
            default_voxel_size,
            room_dimensions,
            protected_bounds=table_protection_bounds,
            fixture_mask=table_fixture_mask,
        )
        print("table voxels stripped:", removed_table_voxels)
        print("table occlusion voxels stripped:", removed_table_occlusion_voxels)

    occlusion_zone_components = []
    occlusion_zone_stats = {
        "candidate_components": 0,
        "selected_components": 0,
        "attached_components": 0,
        "detached_components": 0,
        "attached_voxels": 0,
        "detached_voxels": 0,
        "selected_voxels": 0,
        "skipped_large_components": 0,
    }
    if BUILD_OCCLUSION_ZONE_BOXES or MERGE_OCCLUSIONS_WITH_OBJECTS:
        occlusion_zone_components, occlusion_zone_stats = draft_find_occlusion_zone_components(
            decomposition_grid,
            clean_occlusion_grid,
            min_voxels=OCCLUSION_MIN_COMPONENT_VOXELS,
            connectivity=COMPONENT_CONNECTIVITY,
            include_detached=INCLUDE_DETACHED_OCCLUSION_ZONES,
        )
        print("occlusion zone candidate components:", occlusion_zone_stats["candidate_components"])
        print("occlusion zone components selected:", occlusion_zone_stats["selected_components"])
        print("occlusion zone components attached to visible objects:", occlusion_zone_stats["attached_components"])
        print("occlusion zone detached components kept:", occlusion_zone_stats["detached_components"])
        print("occlusion zone voxels attached:", occlusion_zone_stats["attached_voxels"])
        print("occlusion zone detached voxels kept:", occlusion_zone_stats["detached_voxels"])
        print("occlusion zone large components skipped:", occlusion_zone_stats["skipped_large_components"])

    occlusion_entity_mask = np.zeros_like(decomposition_grid, dtype=bool)
    if MERGE_OCCLUSIONS_WITH_OBJECTS:
        before_merge = int(np.count_nonzero(decomposition_grid))
        occlusion_entity_mask = draft_build_occlusion_entity_mask(
            decomposition_grid,
            occlusion_zone_components,
            COMPONENT_CONNECTIVITY,
        )
        decomposition_grid = np.logical_or(decomposition_grid, occlusion_entity_mask)
        print("occlusion merge voxels added:", int(np.count_nonzero(decomposition_grid)) - before_merge)

    occupied_count = int(np.count_nonzero(decomposition_grid))

    components = draft_find_connected_components(
        decomposition_grid,
        min_voxels=MIN_COMPONENT_VOXELS,
        connectivity=COMPONENT_CONNECTIVITY,
    )
    draft_print_component_stats("decomposition grid", components, occupied_count)

    component_points = [
        draft_component_to_points(component, default_voxel_size, room_dimensions)
        for component in components
    ]
    component_geometries = draft_colorize_components(
        component_points,
        component_indices=components,
        occlusion_mask=occlusion_entity_mask,
    )
    boxes, box_records = draft_build_component_bounding_boxes(
        components,
        default_voxel_size,
        room_dimensions,
        kind="entity",
        occlusion_mask=occlusion_entity_mask,
    )
    print("entity boxes produced:", len(boxes))
    draft_log_box_proxies(box_records)
    hull_meshes = []
    hull_records = []
    hull_chunks = []
    if SHOW_COMPONENT_CONVEX_HULLS:
        hull_meshes, hull_records, hull_chunks = draft_build_component_convex_hulls(
            components,
            default_voxel_size,
            room_dimensions,
            kind="entity",
            margin_m=CONVEX_HULL_MARGIN_M,
        )
        print("entity convex hulls produced:", len(hull_meshes))
        draft_log_convex_hulls(hull_records)

    occlusion_boxes = []
    occlusion_box_records = []
    occlusion_hulls = []
    occlusion_hull_records = []
    occlusion_hull_chunks = []
    occlusion_geometries = []
    if not MERGE_OCCLUSIONS_WITH_OBJECTS and occlusion_zone_components:
        occlusion_boxes, occlusion_box_records = draft_build_component_bounding_boxes(
            occlusion_zone_components,
            default_voxel_size,
            room_dimensions,
            kind="occlusion",
            margin_m=OCCLUSION_BOX_MARGIN_M,
            max_boxes_per_component=OCCLUSION_MAX_BOXES_PER_ZONE,
            color_override=OCCLUSION_COLOR,
            occlusion_mask=draft_occlusion_components_to_mask(occlusion_zone_components, decomposition_grid.shape),
        )
        print("occlusion zone boxes produced:", len(occlusion_boxes))
        draft_log_box_proxies(occlusion_box_records)

        occlusion_points = [
            draft_component_to_points(component, default_voxel_size, room_dimensions)
            for component in occlusion_zone_components
        ]
        occlusion_geometries = draft_colorize_components(
            occlusion_points,
            color_override=OCCLUSION_COLOR,
        )

        if SHOW_COMPONENT_CONVEX_HULLS:
            occlusion_hulls, occlusion_hull_records, occlusion_hull_chunks = draft_build_component_convex_hulls(
                occlusion_zone_components,
                default_voxel_size,
                room_dimensions,
                kind="occlusion",
                margin_m=CONVEX_HULL_MARGIN_M,
                color_override=OCCLUSION_COLOR,
            )
            print("occlusion zone convex hulls produced:", len(occlusion_hulls))

    phantom_geometries = []
    if SHOW_PHANTOM_FREE_SPACE_DIAGNOSTIC:
        print("computing phantom free space diagnostic (Unity GT vs Cameras)...")
        try:
            gt_metadata, gt_raw = azurion_dataset.load_unity_voxels(sample_path, "props")
            gt_grid = azurion_dataset.reshape_unity_voxel_grid(gt_metadata, gt_raw)

            min_x = min(gt_grid.shape[0], clean_occlusion_grid.shape[0])
            min_y = min(gt_grid.shape[1], clean_occlusion_grid.shape[1])
            min_z = min(gt_grid.shape[2], clean_occlusion_grid.shape[2])

            gt_sliced = gt_grid[:min_x, :min_y, :min_z]
            occ_sliced = clean_occlusion_grid[:min_x, :min_y, :min_z]
            obj_sliced = occupied_grid[:min_x, :min_y, :min_z]

            # Phantom voxels: GT says it's an object, but cameras marked it as free space (~occ_sliced).
            phantom_mask = gt_sliced & ~occ_sliced & ~obj_sliced

            phantom_components = draft_find_connected_components(phantom_mask, min_voxels=1, connectivity=COMPONENT_CONNECTIVITY)
            if phantom_components:
                phantom_voxels = sum(len(c) for c in phantom_components)
                print(f"DIAGNOSTIC: Found {phantom_voxels} phantom erased voxels! Rendering in RED.")
                phantom_points = [draft_component_to_points(c, default_voxel_size, room_dimensions) for c in phantom_components]
                phantom_geometries = draft_colorize_components(
                    phantom_points,
                    color_override=PHANTOM_COLOR,
                )
        except Exception as e:
            print(f"Could not load Unity GT for phantom diagnostic: {e}")

    visualization_elements = []
    visualization_elements.extend(component_geometries)
    if SHOW_COMPONENT_CONVEX_HULLS:
        visualization_elements.extend(hull_meshes)
    if SHOW_COMPONENT_BOXES:
        visualization_elements.extend(boxes)
        visualization_elements.extend(occlusion_boxes)
    if SHOW_OCCLUSION_ZONE_BOXES:
        visualization_elements.extend(occlusion_geometries)
        visualization_elements.extend(occlusion_hulls)
    if SHOW_BLIND_SPOTS:
        visualization_elements.extend(blind_spot_geometries)
        visualization_elements.extend(blind_spot_hulls)
    if SHOW_ROBOT_HULL_OVERLAY:
        visualization_elements.extend(robot_model.get_all_meshes())
    if SHOW_PHANTOM_FREE_SPACE_DIAGNOSTIC:
        visualization_elements.extend(phantom_geometries)
    if SHOW_FULL_VOXEL_GRID:
        visualization_elements.append(converted_voxel_grid.voxel_grid)

    return {
        "sample_index": sample_index,
        "sample_path": sample_path,
        "room_dimensions": room_dimensions,
        "room_point_cloud": room_point_cloud.point_cloud,
        "geometries": visualization_elements,
        "components": components,
        "hull_meshes": hull_meshes,
        "hull_records": hull_records,
        "hull_chunks": hull_chunks,
        "boxes": boxes,
        "box_records": box_records,
        "occlusion_zone_components": occlusion_zone_components,
        "occlusion_entity_mask": occlusion_entity_mask,
        "occlusion_boxes": occlusion_boxes,
        "occlusion_box_records": occlusion_box_records,
        "occlusion_hulls": occlusion_hulls,
        "occlusion_hull_records": occlusion_hull_records,
        "occlusion_hull_chunks": occlusion_hull_chunks,
        "occlusion_geometries": occlusion_geometries,
        "blind_spot_boxes": blind_spot_boxes,
        "blind_spot_box_records": blind_spot_box_records,
        "blind_spot_geometries": blind_spot_geometries,
        "blind_spot_hulls": blind_spot_hulls,
        "blind_spot_hull_records": blind_spot_hull_records,
        "blind_spot_hull_chunks": blind_spot_hull_chunks,
        "blind_spot_components": blind_spot_components,
        "occlusion_zone_stats": occlusion_zone_stats,
        "raw_occupied_count": raw_occupied_count,
        "decomposition_occupied_count": occupied_count,
        "occlusion_grid": occlusion_grid,
        "occlusion_not_free": occlusion_not_free,
    }


def draft_update_side_rgb_window(sample_path, sample_index, image_state):
    # this is just a visual reference, not part of the reconstruction pipeline.
    if not SHOW_SIDE_RGB_IMAGE:
        return

    image_path = os.path.join(sample_path, SIDE_RGB_FILE_NAME)
    if not os.path.exists(image_path):
        print(f"side rgb image missing: {image_path}")
        return

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/azurion_matplotlib")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        image = mpimg.imread(image_path)
        title = f"side rgb reference - sample_{sample_index:04d}"

        if image_state.get("figure") is None:
            plt.ion()
            figure, axis = plt.subplots(num="side rgb reference")
            artist = axis.imshow(image)
            axis.axis("off")
            axis.set_title(title)
            manager = getattr(figure.canvas, "manager", None)
            if manager is not None:
                manager.set_window_title(title)
            image_state["figure"] = figure
            image_state["axis"] = axis
            image_state["artist"] = artist
            plt.show(block=False)
        else:
            figure = image_state["figure"]
            axis = image_state["axis"]
            artist = image_state["artist"]
            artist.set_data(image)
            axis.set_title(title)
            manager = getattr(figure.canvas, "manager", None)
            if manager is not None:
                manager.set_window_title(title)
            figure.canvas.draw_idle()

        plt.pause(0.001)
    except Exception as exc:
        print(f"side rgb viewer failed: {exc}")


def draft_show_interactive_sample_browser(dataset, start_index=DEFAULT_SAMPLE_INDEX):
    # right arrow reloads the next sample into the same window.
    state = {
        "sample_index": start_index,
        "geometries": [],
        "side_rgb": {},
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="decomposition draft - right arrow for next sample")

    def load_sample(sample_index):
        for geometry in state["geometries"]:
            vis.remove_geometry(geometry, reset_bounding_box=False)

        result = draft_process_sample(dataset, sample_index)
        state["sample_index"] = sample_index
        state["geometries"] = result["geometries"]
        draft_update_side_rgb_window(result["sample_path"], sample_index, state["side_rgb"])

        for geometry in state["geometries"]:
            vis.add_geometry(geometry, reset_bounding_box=True)

        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0])
        render_option.line_width = 2.0
        print("right arrow: next sample")
        return result

    def on_right_arrow(visualizer):
        next_index = (state["sample_index"] + 1) % len(dataset.List_of_Samples)
        load_sample(next_index)
        return False

    load_sample(start_index)
    vis.register_key_callback(RIGHT_ARROW_KEY, on_right_arrow)
    vis.run()
    vis.destroy_window()


def draft_run_decomposition_smoke_test():
    draft_require_stable_runtime()

    parser = argparse.ArgumentParser(description="Run decomposition smoke test / visualization.")
    parser.add_argument("--rig-id", type=str, default=None, help="Specify the Rig ID (e.g., '4CamClassic').")
    parser.add_argument("--sample", type=int, default=None, help="Index in the dataset OR sample number (if --rig-id is used).")
    parser.add_argument("--show-voxel-grid", action="store_true", help="Include the full Open3D room voxel grid in the visualization.")
    args = parser.parse_args()

    global SHOW_FULL_VOXEL_GRID
    if args.show_voxel_grid:
        SHOW_FULL_VOXEL_GRID = True

    # make the vscode run button behave even if the workspace cwd is the parent unity repo.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    dataset = Dataset()
    if not dataset.List_of_Samples:
        raise RuntimeError(f"no sample folder found in {DepthImageFolder}")

    selected_index = DEFAULT_SAMPLE_INDEX
    if args.sample is not None:
        if args.rig_id is not None:
            target_name = f"sample_{args.sample:04d}"
            selected_idx = next(
                (i for i, s in enumerate(dataset.samples) if s.rig_id == args.rig_id and s.sample_name == target_name),
                None
            )
            if selected_idx is None:
                print(f"Error: Sample '{target_name}' not found for rig '{args.rig_id}'.")
                return
            selected_index = selected_idx
        else:
            if args.sample < 0 or args.sample >= len(dataset.List_of_Samples):
                print(f"Error: Sample index {args.sample} out of bounds (0 to {len(dataset.List_of_Samples)-1}).")
                return
            selected_index = args.sample

    if OPEN_INTERACTIVE_VIEW:
        draft_show_interactive_sample_browser(dataset, start_index=selected_index)
    else:
        result = draft_process_sample(dataset, selected_index)
        if DISPLAY_OCCLUSION_GRID:
            print("occlusion zone component summary because DISPLAY_OCCLUSION_GRID is true...")
            draft_print_component_stats(
                "occlusion zone grid",
                result["occlusion_zone_components"],
                result["occlusion_zone_stats"]["attached_voxels"],
            )
        print("interactive view skipped because DECOMPOSITION_DRAFT_NO_VIEW=1")


if __name__ == "__main__":
    draft_run_decomposition_smoke_test()
