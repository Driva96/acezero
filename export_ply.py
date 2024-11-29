from typing import Union, Tuple, Dict

from pathlib import Path
from typing import List, Any 
import numpy as np

import pandas as pd

import pycolmap
from scipy.spatial.transform import Rotation

import open3d as o3d

import ast

def to_np(df: pd.DataFrame, columns: List[str]):
    for c in columns:
        df[c] = df[c].apply(val_to_np)
    return df

def val_to_np(x):
    if isinstance(x, str):
        return np.array(ast.literal_eval(x))
    else:
        return np.array(x)

def write_ply(
        colmap_file_name: Path,
        fused_file_name: Path,
        points3D: Dict[int, pycolmap.Point3D],
        objects: Union[pd.DataFrame],
        color = np.array([255,0,0], np.uint8)):
    _dtype = np.dtype([
        ('x', 'f4'),       # float32 for x coordinate
        ('y', 'f4'),       # float32 for y coordinate
        ('z', 'f4'),       # float32 for z coordinate
        ('red', 'u1'),     # uint8 for red color component
        ('green', 'u1'),   # uint8 for green color component
        ('blue', 'u1')     # uint8 for blue color component
    ])

    if isinstance(objects, pd.DataFrame):
        to_np(objects, ['point3D'])
    
    # Replace NaN in 'point3D' with np.empty
    objects['point3D'] = objects['point3D'].apply(lambda x: np.empty((0, 3)) if (isinstance(x, float) and np.isnan(x)) or isinstance(x, pd._libs.missing.NAType) else x)

    # Stack the values
    objects = np.vstack(objects['point3D'].tolist()) if not objects.empty else np.empty((0, 3))

    ply_obj = ply.PlyData()

    point_len = len(points3D)
    points = np.empty(len(points3D) + len(objects), dtype=_dtype)
    for i, (_, p) in enumerate(points3D.items()):
        x,y,z = p.xyz.astype(np.float32)
        r,g,b = p.color.astype(np.uint8)
        points['x'][i] = x
        points['y'][i] = y
        points['z'][i] = z
        points['red'][i] = r
        points['green'][i] = g
        points['blue'][i] = b
    
    ply_obj.elements = [ply.PlyElement.describe(points, 'vertex')]
    ply_obj.write(str(colmap_file_name))
    
    for i, p in enumerate(objects):
        x,y,z = p.astype(np.float32)
        r,g,b = color
        i += point_len
        points['x'][i] = x
        points['y'][i] = y
        points['z'][i] = z
        points['red'][i] = r
        points['green'][i] = g
        points['blue'][i] = b

    ply_obj.elements = [ply.PlyElement.describe(points, 'vertex')]
    ply_obj.write(str(fused_file_name))

def read_pandas(path: Path)->pd.DataFrame:
    names = ['filename', 'qw', 'qx', 'qy', 'qz', 'x', 'y', 'z', 'focal_length', 'confidence']
    df = pd.read_csv(path.resolve(strict=True),header=None, names=names, sep='\s+')
    return df

def _convert_cv_to_gl(pose):
        """
        Convert a pose from OpenCV to OpenGL convention (and vice versa).

        @param pose: 4x4 camera pose.
        @return: 4x4 camera pose.
        """
        gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
        return gl_to_cv * pose

def get_3d_point(row: pd.Series):
    # read pose
    q_wxyz = row.iloc[1:5]
    t_xyz = row.iloc[5:8]

    # quaternion to rotation matrix
    R = Rotation.from_quat(
            q_wxyz[1:].append(pd.Series([q_wxyz[0]]), ignore_index=True)
        ).as_matrix()


    # construct full pose matrix
    T_world2cam = np.eye(4)
    T_world2cam[:3, :3] = R
    T_world2cam[:3, 3] = t_xyz

    # pose files contain world-to-cam but we need cam-to-world
    T_cam2world = np.linalg.inv(T_world2cam)
    # T_cam2world_opengl = _convert_cv_to_gl(T_cam2world)

    confidence_threshold = 1000

    # intrinsics
    K = np.identity(3)

    K[0, 0] = row['focal_length']
    K[1, 1] = row['focal_length']
    K[0, 2] = 640
    K[1, 2] = 360

    if row.loc['confidence'] > 0:
        geom = draw_camera(K, T_cam2world[:3,:3], T_cam2world[:3,-1], 1280, 720)
        return geom
    else:
        return np.NaN

def draw_camera(K, R, t, w, h, scale=1, color=[0.8, 0.2, 0.8]) -> List[Any]:
    """Create axis, plane and pyramed geometries in o3d format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale
    )
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    print(f'w: {width}, height: {height} \n')
    plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]

def show(__vis):
    __vis.poll_events()
    __vis.update_renderer()
    __vis.run()
    __vis.destroy_window()

def main():
    df = read_pandas(Path('results/poses_final.txt'))
    df['geometry'] = df.apply(get_3d_point, axis=1)

    # Initialize an empty geometry to combine all geometries
    #combined_geometry = o3d.geometry.TriangleMesh()

    geometries = []
    for index, row in df.iterrows():
        if isinstance(row['geometry'], (list, np.ndarray, pd.Series)):
            # If 'geometry' is a list, numpy array, or pandas Series, check if any element is NaN
            if not pd.isna(row['geometry']).any():
                # No NaN values in the array
                geometries.append(row['geometry'])
        else:
            # If 'geometry' is a scalar value, just check if it is NaN
            if not pd.isna(row['geometry']):
                # No NaN value in the scalar
                pass
    
    # Assuming you want to combine all geometries into a single mesh
    combined_geometry = o3d.geometry.TriangleMesh()
    combined_lines = o3d.geometry.LineSet()
    for geom in geometries:
        combined_geometry += (geom[0] + geom[1])
        combined_lines += geom[2]

    # Save the combined geometry as a PLY file
    # Write the combined geometry to a PLY file
    o3d.io.write_triangle_mesh("combined_output.ply", combined_geometry)
    # Save the line set in a different file format (e.g., .off)
    o3d.io.write_line_set("line_set.ply", combined_lines)

if __name__ == '__main__':
    main()