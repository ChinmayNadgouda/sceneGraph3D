#import rosbag
#import cv2
#from cv_bridge import CvBridge
import numpy as np
import os
import open3d as o3d
from open3d import pipelines
from scipy.spatial.transform import Rotation as R
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# Extract translations from poses
def get_translations(poses):
    translations = np.array([pose[:3, 3] for pose in poses])
    return translations


# Extract rotation matrices from poses
def get_rotations(poses):
    rotations = np.array([pose[:3, :3] for pose in poses])
    return rotations
#
#
# # Initialize cv_bridge
# bridge = CvBridge()
#
# # Path to your ROS bag file
# bag_file = '20240924_101254.bag'
#
# # Output directories
# # rgb_dir = 'final_op_small/rgb/'
# # depth_dir = 'final_op_small/depth/'
# #pose_file = 'replica1/poses.txt'
# #pose_file_hybrid = 'replica1/poses_h.txt'
#
# # Create directories if they don't exist
# #os.makedirs('replica1', exist_ok=True)
# # os.makedirs(depth_dir, exist_ok=True)
#
# intrinsic_matrix_temp = np.array([[910.557, 0.0, 650.265],
#                             [0.0, 910.094, 358.224],
#                             [0.0, 0.0, 1.0]])  # Example for PrimeSense camera
#
# def compute_pose(prev_img, curr_img, K):
#     # Convert images to grayscale
#     prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
#
#     # Detect ORB features and compute descriptors
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(prev_gray, None)
#     kp2, des2 = orb.detectAndCompute(curr_gray, None)
#
#     # Match features using BFMatcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#
#     # Sort matches by distance
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     # Extract location of good matches
#     pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#
#     # Find essential matrix
#     E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#
#     # Recover pose from essential matrix
#     _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
#
#     return R, t
#
# # Function to estimate pose between two RGB-D frames
# def estimate_odometry(source_color, source_depth, target_color, target_depth, v):
#     # examples/Python/Basic/rgbd_odometry.py
#     # Define camera intrinsic parameters
#     odo_init = np.eye(4)
#     pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
#     pinhole_camera_intrinsic.set_intrinsics(width=1200, height=680, fx=600, fy=600, cx=599.5, cy=339.5)
#     #pinhole_camera_intrinsic.set_intrinsics(width=1280, height=720, fx=214.313, fy=214.313, cx=214.517, cy=118.207)
#     #pinhole_camera_intrinsic.set_intrinsics(width=1280, height=720, fx=910.557, fy=910.094, cx=650.265, cy=358.224)
#
#     source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         source_color, source_depth)
#     target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         target_color, target_depth)
#     target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#         target_rgbd_image, pinhole_camera_intrinsic)
#
#     option = pipelines.odometry.OdometryOption()
#
#
#
#     [success_color_term, trans_color_term,
#      info] = pipelines.odometry.compute_rgbd_odometry(
#          source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
#          odo_init, pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
#     [success_hybrid_term, trans_hybrid_term,
#      info] = pipelines.odometry.compute_rgbd_odometry(
#          source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
#          odo_init, pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
#     print(trans_hybrid_term)
#     return trans_color_term, trans_hybrid_term
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def extract_translation_rotation(T):
    """Extract translation and rotation (Euler angles) from a 4x4 pose matrix."""
    t = T[:3, 3]  # Translation vector (x, y, z)
    R_matrix = T[:3, :3]  # 3x3 rotation matrix
    #euler_angles = R.from_matrix(R_matrix).as_euler('xyz', degrees=True)  # Convert rotation matrix to Euler angles
    return t, 0
def rotate_axes(R_matrix, rotate_y=90, rotate_z=90):
    """Rotate the y (green) and z (blue) axes by specified angles."""
    # Create rotation matrices for rotating around x, y, and z
    rot_y = R.from_euler('y', rotate_y, degrees=True).as_matrix()  # Rotation around y-axis
    rot_z = R.from_euler('z', rotate_z, degrees=True).as_matrix()  # Rotation around z-axis
    
    # Apply the rotations to the existing rotation matrix
    R_new = np.dot(R_matrix, rot_z)  # Rotate z-axis first
    R_new = np.dot(R_new, rot_y)     # Rotate y-axis after
    
    return R_new
def rotate_y_axis_only(R_matrix, rotate_y=180):
    """Rotate only the y-axis (green axis) by a specified angle."""
    # Create rotation matrix for rotating around y-axis
    rot_y = R.from_euler('y', rotate_y,degrees = True).as_matrix()  # Rotation around y-axis

    # Apply the rotation only to the y-axis (second column of the rotation matrix)
    R_matrix[:, 1] = np.dot(R_matrix[:, 1], rot_y)  # Rotate the y-axis column

    return R_matrix
def rotate_axes_z(R_matrix, rotate_z=180):
    """Rotate the y (green) and z (blue) axes by specified angles."""
    # Create rotation matrices for rotating around x, y, and z
    #rot_y = R.from_euler('y', rotate_y, degrees=True).as_matrix()  # Rotation around y-axis
    rot_z = R.from_euler('z', rotate_z, degrees=True).as_matrix()  # Rotation around z-axis
    
    # Apply the rotations to the existing rotation matrix
    R_new = np.dot(R_matrix, rot_z)  # Rotate z-axis first
    #R_new = np.dot(R_new, rot_y)     # Rotate y-axis after
    
    return R_new

def visualize_trajectory_and_orientation(poses,poses2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    translations = []
    translations2 = []
    orientations = []
    orientations2 = []
    

    # Extract translations and orientations from each pose
    for T in poses:
        t, euler_angles = extract_translation_rotation(T)
        translations.append(t.tolist())
        orientations.append(euler_angles)
    print(translations)
    translations = np.array(translations)
    """
    for T in poses2:
        t, euler_angles = extract_translation_rotation(T)
        translations2.append(t)
        orientations2.append(euler_angles)

    translations2 = np.array(translations2)

    # Plot trajectory (translation vectors)
    #ax.plot(-translations[:, 0], -translations[:, 2],translations[:, 1], label="Trajectory", color='blue', linewidth=2)
    
    ax.plot(translations[:, 0],translations[:, 1],translations[:, 2], label="Trajectory", color='blue', linewidth=2)

    #Visualize orientation at each pose with quivers
    #Visualize orientation at each pose with quivers
    
    for i, T in enumerate(poses2):
        t, _ = extract_translation_rotation(T)
        print(_)
        R_matrix = T[:3, :3]
        
        #R_matrix = rotate_y_axis_only(R_matrix)
        #R_matrix = rotate_axes(R_matrix_n, 90, 270)
    
        # Draw the local coordinate axes at each pose
        if(i == 0):
            ax.quiver(translations2[i, 0],translations2[i, 1],translations2[i, 2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='red', length=0.3,
                      normalize=True)
            ax.quiver(translations2[i, 0],translations2[i, 1],translations2[i, 2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='yellow', length=0.3,
                      normalize=True)
            ax.quiver(translations2[i, 0],translations2[i, 1],translations2[i, 2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='pink', length=0.3,
                      normalize=True)
            break
        else:
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='red', length=0.003,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='green', length=0.003,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='blue', length=0.003,
                      normalize=True)
    """
    ax.plot(translations[:, 0],translations[:, 1],translations[:, 2], label="Trajectory", color='blue', linewidth=2)
    for i, T in enumerate(poses):
        t, _ = extract_translation_rotation(T)
        print(_)
        R_matrix = T[:3, :3]
        
        #R_matrix = rotate_y_axis_only(R_matrix)
        #R_matrix = rotate_axes(R_matrix_n, 90, 270)
    
        # Draw the local coordinate axes at each pose
        if(i == 0):
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='black', length=0.1,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='green', length=0.1,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='blue', length=0.1,
                      normalize=True)      

        else:
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='red', length=0.3,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='green', length=0.3,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='blue', length=0.3,
                      normalize=True)

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory and Orientation Visualization')

    plt.legend()
    plt.show()


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
     """Convert a quaternion into a 3x3 rotation matrix."""
     rotation_matrix = np.array([
         [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
         [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
         [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
     ])
     return rotation_matrix

def create_transformation_matrix(quaternion_translation):
     """Create a 4x4 transformation matrix from a quaternion and translation."""
     qw, qx, qy, qz, tx, ty, tz = quaternion_translation
     q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)

     # Normalize the quaternion
     magnitude = torch.norm(q)

     n_q = q / magnitude
     N_qx, N_qy, N_qz, N_qw = n_q[0], n_q[1], n_q[2], n_q[3]

     # Step 1: Convert quaternion to rotation matrix
     rotation_matrix = quaternion_to_rotation_matrix(N_qw, N_qx, N_qy, N_qz)
     rotation = R.from_quat(q).as_matrix()

     # Step 2: Create a 4x4 transformation matrix
     transformation_matrix = np.eye(4)  # Initialize a 4x4 identity matrix
     transformation_matrix[:3, :3] = rotation  # Set the 3x3 rotation matrix
     transformation_matrix[:3, 3] = [tx, ty, tz]  # Set the translation vector

     return transformation_matrix
#
def matrix_to_string(matrix):
     """Convert a 4x4 matrix to a whitespace-separated string of 16 elements."""
     flat_matrix = matrix.flatten()  # Flatten the matrix into a 1D array
     flat_list = flat_matrix.tolist()  # Convert to list
     string_representation = ' '.join(map(str, flat_list))  # Convert list to a string

     return string_representation
#
def plot_camera_trajectory(poses, scale=0.01):
    # Create an empty Open3D geometry set to hold all elements
    geometries = []

    for i, pose in enumerate(poses):
         # Step 1: Extract the camera position from the pose matrix
         camera_position = pose[:3, 3]

         # Step 2: Extract the camera orientation (rotation)
         # The camera's forward direction is typically along the negative z-axis in camera coordinates
         forward_direction = pose[:3, 2]  # This gives the camera's forward vector in world coordinates

         # Step 3: Create a small coordinate frame to represent the camera pose
         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
             size=scale, origin=camera_position
         )
         camera_frame.rotate(pose[:3, :3])  # Rotate the frame to match the camera orientation

         # Step 4: Add the camera pose frame to the geometry list
         geometries.append(camera_frame)

         # Optionally, plot the trajectory by connecting positions with lines
         if i > 0:
             trajectory_line = o3d.geometry.LineSet(
                 points=o3d.utility.Vector3dVector([previous_camera_position, camera_position]),
                 lines=o3d.utility.Vector2iVector([[0, 1]])
             )
             geometries.append(trajectory_line)

         # Save the current camera position as the previous for the next iteration
         previous_camera_position = camera_position

    # Step 5: Visualize the camera trajectory and orientations
    o3d.visualization.draw_geometries(geometries)
 # Open the file to store pose information
 
open3d = '/home/gokul/Downloads/open3d_example.json'
import json
justpose = True
if justpose:
     poses = []
     pose_folder = '/home/gokul/ConceptGraphs/datasets/Replica/room0/nerf.txt'
     pose_path = '/home/gokul/ConceptGraphs/datasets/Replica/room0/tj60.txt'
     #pose_folder = '/home/gokul/Downloads/open3d_example.log'
     camera_int = '/home/gokul/ConceptGraphs/datasets/Replica/60/dense/0/sparse/cameras.txt'
     dict1 = {}
     dict2 = {}
     int_dict = {}
     open3d_pose = []
     final_int_dict = {}
     colmap = True
     replica = True
     if pose_folder.endswith("nerf.txt"):
        poses = []
        with open(pose_folder, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
     elif pose_folder.endswith(".log"):
                # print("Loading poses from .log format")
                poses_1 = []
                lines = None
                with open(pose_folder, "r") as f:
                    lines = f.readlines()
                if len(lines) % 5 != 0:
                    raise ValueError(
                        "Incorrect file format for .log odom file "
                        "Number of non-empty lines must be a multiple of 5"
                    )
                num_lines = len(lines) // 5
                for i in range(0, num_lines):
                    _curpose = []
                    _curpose.append(list(map(float, lines[5 * i + 1].split())))
                    _curpose.append(list(map(float, lines[5 * i + 2].split())))
                    _curpose.append(list(map(float, lines[5 * i + 3].split())))
                    _curpose.append(list(map(float, lines[5 * i + 4].split())))
                    _curpose = np.array(_curpose).reshape(4, 4)
                    poses_1.append(_curpose)
     elif(colmap):
         with open(pose_folder, 'r') as file, open(camera_int, 'r') as file2:
             lines = file.readlines()
             lines2 = file2.readlines()
             count_int = 0
             for line in lines2:
                 if count_int > 2: 
                     elements_list = line.split()
                     #print(199,int(elements_list[0]))
                     int_dict[int(elements_list[0])] = elements_list[4:]
                 count_int += 1
             #poses = []
             count = 0
             for line in lines:
                 if (line.__contains__('.jpg') or line.__contains__('.JPG')):
                     count += 1   
                     elements_list = line.split()
                     image_id = int(elements_list[0])
                     #if(image_id in int_dict.keys()):
                     #   final_int_dict[image_id] = int_dict[image_id+550]
                     #id_temp = elements_list[-1].split('/')[1]
                     #id = id_temp.split('.')[0]
                     id = elements_list[-1].split('.')[0]
                     #id = id_temp.split('.')[0]
                     dict1[id] = []

                     # Step 2: Convert the list of strings to a list of integers
                     elements_list = list(map(float, elements_list[1:8]))
                     transformation_matrix = create_transformation_matrix(elements_list)
                     dict2[id] = transformation_matrix
                     # Convert the 4x4 matrix to a string of 16 elements whitespace-separated
                     matrix_string = matrix_to_string(transformation_matrix)
                     dict1[id] = matrix_string
             # Write the dictionary to a file (JSON format)

             
             sorted_int = {key: int_dict[key]  for key in sorted(final_int_dict.keys())}
             sorted_dict = {key: dict2[key].tolist()  for key in dict2.keys()}
             sorted_dict_nd = {key: dict2[key]  for key in sorted(dict2.keys())}
             poses_1 = dict2.values()
             print(dict2)
             #plot_camera_trajectory(poses_1)
             with open('/home/gokul/Downloads/output__.json', 'w') as file:
                 json.dump(sorted_dict, file, indent=4)
             with open('/home/gokul/ConceptGraphs/datasets/Replica/60/dense/0/sparse/int.json', 'w') as file:
                 json.dump(sorted_int, file, indent=4)   
             print(poses_1)
     
         poses = []
         with open(pose_path, "r") as f:
            lines = f.readlines()

         for i in range(len(lines)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            #c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
#print(poses)
"""
Q = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
poses_1 = torch.tensor(np.array(list(poses_1)), dtype=torch.double) 

    
    
P_gt_inv = np.linalg.inv(np.array(list(poses)[0]))   
T_off =   np.dot(P_gt_inv, np.array(list(poses_1)[0]))
print('t_off', T_off) 
print(list(poses)[0])

print(np.dot(T_off, np.array(list(poses_1)[0])))
print(list(poses_1)[0])
for pose in poses_1:   
    pose = torch.tensor(T_off) @ pose
"""
#visualize_trajectory_and_orientation(poses_1,poses)
visualize_trajectory_and_orientation(poses,poses)


# elif (replica):
#     print('ithe')
#     # Define the folder paths
#     rgb_folder = '/home/student/Data/output99/rgb'
#     depth_folder = '/home/student/Data/output99/depth'
#     prev_rgb_filename = None
#     prev_resized_depth_filename = None
#     world_poses = []
#     pose = np.identity(4)
#     # List all the image files in both folders (assuming both have the same filenames)
#     rgb_files = sorted(os.listdir(rgb_folder))
#     depth_files = sorted(os.listdir(depth_folder))
#
#     # Ensure both folders have the same number of files
#     if len(rgb_files) != len(depth_files):
#         raise ValueError("The number of RGB and Depth images should be the same.")
#
#     # Loop through each image pair
#     with open(pose_file, 'w') as pf, open(pose_file_hybrid, 'w') as pf_h:
#         count = 0
#         for rgb_file, depth_file in zip(rgb_files, depth_files):
#             rgb_path = os.path.join(rgb_folder, rgb_file)
#             depth_path = os.path.join(depth_folder, depth_file)
#
#             # if prev_resized_depth_filename is not None and prev_rgb_filename is not None:
#             #             print(prev_rgb_filename, rgb_path)
#             #             print(prev_resized_depth_filename, depth_path)
#             #             # Estimate the transformation between the current and previous frames
#             #             source_color = o3d.io.read_image(prev_rgb_filename)
#             #             source_depth = o3d.io.read_image(prev_resized_depth_filename)
#             #             target_color = o3d.io.read_image(rgb_path)
#             #             target_depth = o3d.io.read_image(depth_path)
#
#             #             pose, pose_h = estimate_odometry(source_color, source_depth, target_color, target_depth,pose)
#             #             if pose is not None:
#             #                 world_poses.append(pose)
#             #             #print(pose)
#
#             #             if pose is not None:
#             #                 temp_pose = ''
#             #                 for poses in pose:
#             #                     temp_pose = temp_pose + ' ' +' '.join(map(str, poses)).strip()
#             #                 # Save the pose (4x4 matrix) with the timestamp
#             #                 temp_pose_h = ''
#             #                 for poses in pose_h:
#             #                     temp_pose_h = temp_pose_h + ' ' +' '.join(map(str, poses)).strip()
#             #                 pf.write(temp_pose.strip() +'\n')
#             #                 pf_h.write(temp_pose_h.strip() +'\n')
#             #             else:
#             #                 print(f"Odometry failed between frames  {count}.")
#
#             # Update previous frame information
#             prev_rgb_filename = rgb_path
#             #prev_depth_filename = depth_filename
#             prev_resized_depth_filename = depth_path
#
#             print(count)
#             count += 1
#         #plot_camera_trajectory(world_poses)
# elif replica_orb:
#
#     # Camera intrinsic parameters (replace with your calibration values)
#     #fx, fy, cx, cy = 910.557, 910.094, 650.265, 358.224
#     fx=600
#     fy=600
#     cx=599.5
#     cy=339.5
#     K = np.array([[fx, 0, cx],
#                   [0, fy, cy],
#                   [0, 0, 1]])
#
#     # Directory containing the RGB images
#     image_dir = "/home/student/Data/replica/rgb/"
#     image_files = sorted(os.listdir(image_dir))
#
#     # Initialize pose list
#     poses = []
#     current_pose = np.eye(4)  # Initial pose as identity matrix
#
#     # Iterate through images
#     prev_img = None
#     for image_file in image_files:
#         curr_img = cv2.imread(os.path.join(image_dir, image_file))
#
#         if prev_img is not None:
#             # Compute pose between previous and current images
#             R, t = compute_pose(prev_img, curr_img, K)
#
#             # Update the current pose
#             current_pose = current_pose @ np.block([[R, t], [0, 0, 0, 1]])
#             poses.append(current_pose)
#             print(current_pose)
#
#         prev_img = curr_img
#     plot_camera_trajectory(poses)
# else:
#     print('here')
#     rgb_dir = '/home/student/Data/op99/results'
#     # depth_dir = '/home/student/Data/op99/depth'
#     # os.makedirs(depth_dir, exist_ok=True)
#     os.makedirs(rgb_dir, exist_ok=True)
#     if 1:
#     #with open(pose_file, 'w') as pf, open(pose_file_hybrid, 'w') as pf_h:
#
#         # Variables to store the previous frame
#         prev_rgb = None
#         prev_depth = None
#         prev_timestamp = None
#         camera_pose = np.eye(4)
#         world_poses = [camera_pose]
#         pose = np.identity(4)
#
#         # Open the bag file
#         with rosbag.Bag('/home/student/Data/' + bag_file, 'r') as bag:
#             count = 0
#             for topic, msg, t in bag.read_messages():
#                 timestamp = str(t.to_nsec())
#
#                 # Extract RGB images
#
#                 if topic == '/device_0/sensor_0/Depth_0/image/data':  # Adjust topic name if different
#                     depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#                     depth_filename = os.path.join(rgb_dir, f'depth_{count}.png')
#
#
#                 # Extract Depth images
#                 elif topic == '/device_0/sensor_1/Color_0/image/data':  # Adjust topic name if different
#                     rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#                     rgb_filename = os.path.join(rgb_dir, f'rgb_{count}.png')
#                     rgb_height, rgb_width = rgb_image.shape[:2]
#                     dep_height, dep_width = depth_image.shape[:2]
#                     # Resize the depth image to match the RGB image dimensions
#                     resized_depth_image = cv2.resize(depth_image, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
#                     resized_depth_filename = os.path.join(rgb_dir, f'depth_{count}_resized.png')
#
#                     resized_col_image = cv2.resize(rgb_image, (dep_width, dep_height), interpolation=cv2.INTER_NEAREST)
#                     resized_col_filename = os.path.join(rgb_dir, f'rgb_{count}.png')
#                     cv2.imwrite(depth_filename, depth_image)
#                     cv2.imwrite(resized_col_filename, resized_col_image)
#
#
#                     # # If previous frame exists, estimate pose
#                     # if prev_rgb is not None and prev_depth is not None:
#                     #     # Estimate the transformation between the current and previous frames
#                     #     source_color = o3d.io.read_image(prev_rgb_filename)
#                     #     source_depth = o3d.io.read_image(prev_resized_depth_filename)
#                     #     target_color = o3d.io.read_image(rgb_filename)
#                     #     target_depth = o3d.io.read_image(resized_depth_filename)
#
#                     #     pose, pose_h = estimate_odometry(source_color, source_depth, target_color, target_depth,pose)
#                     #     if pose is not None:
#                     #         world_poses.append(pose)
#                     #     #print(pose)
#
#                     #     if pose is not None:
#                     #         temp_pose = ''
#                     #         for poses in pose:
#                     #             temp_pose = temp_pose + ' ' +' '.join(map(str, poses)).strip()
#                     #         # Save the pose (4x4 matrix) with the timestamp
#                     #         temp_pose_h = ''
#                     #         for poses in pose_h:
#                     #             temp_pose_h = temp_pose_h + ' ' +' '.join(map(str, poses)).strip()
#                     #         pf.write(temp_pose.strip() +'\n')
#                     #         pf_h.write(temp_pose_h.strip() +'\n')
#                     #     else:
#                     #         print(f"Odometry failed between frames {prev_timestamp} and {timestamp}.")
#
#                     # # Update previous frame information
#                     # prev_rgb_filename = rgb_filename
#                     # #prev_depth_filename = depth_filename
#                     # prev_resized_depth_filename = resized_depth_filename
#                     # prev_rgb = rgb_image
#                     # prev_depth = depth_image
#                     # prev_timestamp = timestamp
#                     # prev_count = count
#                     print(count)
#                     count += 1
#
#             # plot_camera_trajectory(world_poses)
#
#

if False:
    import pyrealsense2  as rs
    fname = '20240924_101254.bag'
    frame_present = True
    frameset = []
    count = 0
    d = rs.decimation_filter()
    cfg = rs.config()
    cfg.enable_device_from_file(fname, repeat_playback=False)
    # setup pipeline for the bag file
    pipe = rs.pipeline()
    # start streaming from file
    profile = pipe.start(cfg)
    while frame_present:
        try:
            frame = pipe.wait_for_frames()
            frame.keep()
            frameset.append(frame)
            count += 1
            if count == 30:
                break
        except RuntimeError:
            print("number of frames extracted:", str(count))
            frame_present = False
            continue
    pipe.stop()

    # set alignment
    alignedset = []
    align_to = rs.stream.depth
    for f in frameset:
        align = rs.align(align_to)
        aligned = align.process(f)
        depth = f.get_depth_frame()
        image = f.get_color_frame()

        processed = d.process(image)
        prof = processed.get_profile().as_video_stream_profile()
        extr_depth_to_colr = depth.profile.get_extrinsics_to(image.profile)
        intr = prof.get_intrinsics()
        print(extr_depth_to_colr)


# import cv2
# import numpy as np
# import os

# def compute_pose(prev_img, curr_img, K):
#     # Convert images to grayscale
#     prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

#     # Detect ORB features and compute descriptors
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(prev_gray, None)
#     kp2, des2 = orb.detectAndCompute(curr_gray, None)

#     # Match features using BFMatcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)

#     # Sort matches by distance
#     matches = sorted(matches, key=lambda x: x.distance)

#     # Extract location of good matches
#     pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#     # Find essential matrix
#     E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

#     # Recover pose from essential matrix
#     _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

#     return R, t

# # Camera intrinsic parameters (replace with your calibration values)
# fx, fy, cx, cy = 910.557, 910.094, 650.265, 358.224
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])

# # Directory containing the RGB images
# image_dir = "/home/student/Data/output3/rgb/"
# image_files = sorted(os.listdir(image_dir))

# # Initialize pose list
# poses = []
# current_pose = np.eye(4)  # Initial pose as identity matrix

# # Iterate through images
# prev_img = None
# for image_file in image_files:
#     curr_img = cv2.imread(os.path.join(image_dir, image_file))
    
#     if prev_img is not None:
#         # Compute pose between previous and current images
#         R, t = compute_pose(prev_img, curr_img, K)
        
#         # Update the current pose
#         current_pose = current_pose @ np.block([[R, t], [0, 0, 0, 1]])
#         poses.append(current_pose)

#     prev_img = curr_img

# # Print the estimated poses
# for i, pose in enumerate(poses):
#     print(f"Pose {i}:")
#     print(pose)

# with open("output3/posee.txt", 'w') as pf:
#     for pose in poses:
#         temp_pose = ''
#         for p in pose:
#             temp_pose = temp_pose + ' ' +' '.join(map(str, p)).strip() 
#             # Save the pose (4x4 matrix) with the timestamp
#         pf.write(temp_pose+'\n')gokul/ConceptGraphs/datasets/0
