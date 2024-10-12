import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from natsort import natsorted
import glob

def pose_to_transformation_matrix(quaternion, translation):
    # Convert quaternion (QW, QX, QY, QZ) to a rotation matrix
    r = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # (QX, QY, QZ, QW)
    rotation_matrix = r.as_matrix()  # 3x3 rotation matrix

    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix
def transform_point_cloud(pcd, transformation_matrix):
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)

    # Apply the transformation to the points (homogeneous coordinates)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points_homogeneous = np.dot(transformation_matrix, points_homogeneous.T).T

    # Extract the 3D coordinates from the transformed points
    transformed_points = transformed_points_homogeneous[:, :3]

    # Create a new Open3D point cloud with transformed points
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    return transformed_pcd
def create_point_cloud_from_depth_image(depth_image_path, rgb_image_path, camera_intrinsics):
    # Load depth and RGB images
    print(rgb_image_path)
    depth_image = o3d.io.read_image(depth_image_path)
    rgb_image = o3d.io.read_image(rgb_image_path)
    
    if depth_image is None or rgb_image is None:
        print("Error: Unable to load depth or RGB image.")
        return None

    # Create an RGBD image from the depth and color images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.io.read_image(rgb_image_path), 
        o3d.io.read_image(depth_image_path), 
        convert_rgb_to_intensity=True
    )

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        camera_intrinsics
    )
    return pcd
    
def get_colmap_poses(pose_folder):
    with open(pose_folder, 'r') as file:
             lines = file.readlines()
             count = 0
             colmap_poses = []
             temp_dict = {}
             for line in lines:
                 if (line.__contains__('.jpg') or line.__contains__('.JPG')):
                     count += 1   
                     elements_list = line.split()
                     #id_temp = elements_list[-1].split('/')[1]
                     #id = id_temp.split('.')[0]
                     # Step 2: Convert the list of strings to a list of integers
                     id = elements_list[-1].split('.')[0]
                     elements_list = list(map(float, elements_list[1:8]))
                     temp_dict[id] = {}
                     temp_dict[id]['quaternion'] = elements_list[0:4]
                     temp_dict[id]['translation'] = elements_list[4:]
             sorted_poses = {key: temp_dict[key]  for key in sorted(temp_dict.keys())}
             colmap_poses.append(list(sorted_poses.values()))
    return colmap_poses
    
def get_paths(input_folder):
    rgb_image_paths = natsorted(glob.glob(f"{input_folder}/results/frame*.jpg"))
    depth_image_paths = natsorted(glob.glob(f"{input_folder}/results/depth*.png"))
    
    return rgb_image_paths, depth_image_paths

rgb_image_paths, depth_image_paths = get_paths('/home/gokul/ConceptGraphs/datasets/Replica/room0')
num_images = len(rgb_image_paths)
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsics.set_intrinsics(width=1200, height=680, fx=600, fy=600, cx=599.5, cy=339.5)

for i in range(num_images):
    colmap_poses = get_colmap_poses('/home/gokul/ConceptGraphs/datasets/Replica/60/sparse/0/images.txt')[0]
    quaternion = colmap_poses[i]["quaternion"]  # [QW, QX, QY, QZ]
    translation = colmap_poses[i]["translation"]  # [TX, TY, TZ]
    
    # Convert pose to transformation matrix
    transformation_matrix = pose_to_transformation_matrix(quaternion, translation)
    
    # Create point cloud from depth map
    pcd = create_point_cloud_from_depth_image(depth_image_paths[i], rgb_image_paths[i], camera_intrinsics)
    # Transform point cloud using the camera pose
    transformed_pcd = transform_point_cloud(pcd, transformation_matrix)
    
    # Combine or visualize the point cloud
    # (You can append multiple point clouds if necessary)
    
    o3d.visualization.draw_geometries([transformed_pcd])
