import torch
import numpy as np
#import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

pose_path = '/home/gokul/ConceptGraphs/datasets/Replica/room0/tj60.txt'
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

def load_poses():
    # Determine whether the posefile ends in ".log"
    # a .log file has the following format for each frame
    # frame_idx frame_idx+1
    # row 1 of 4x4 transform
    # row 2 of 4x4 transform
    # row 3 of 4x4 transform
    # row 4 of 4x4 transform
    # [repeat for all frames]
    #
    # on the other hand, the "poses_o3d.txt" or "poses_dvo.txt" files have the format
    # 16 entries of 4x4 transform
    # [repeat for all frames]
    if pose_path.endswith(".log"):
        # print("Loading poses from .log format")
        poses = []
        lines = None
        with open(pose_path, "r") as f:
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
            poses.append(_curpose)
    else:
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
    return poses


poses = load_poses()
print(poses)
#plot_camera_trajectory(poses)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def extract_translation_rotation(T):
    """Extract translation and rotation (Euler angles) from a 4x4 pose matrix."""
    t = T[:3, 3]  # Translation vector (x, y, z)
    R_matrix = T[:3, :3]  # 3x3 rotation matrix
    euler_angles = R.from_matrix(R_matrix).as_euler('xyz', degrees=True)  # Convert rotation matrix to Euler angles
    return t, euler_angles


def rotate_y_axis_only(R_matrix, rotate_y=90):
    """Rotate only the y-axis (green axis) by a specified angle."""
    # Create rotation matrix for rotating around y-axis
    rot_y = R.from_euler('y', rotate_y).as_matrix()  # Rotation around y-axis

    # Apply the rotation only to the y-axis (second column of the rotation matrix)
    R_matrix[:, 1] = np.dot(R_matrix[:, 1], rot_y)  # Rotate the y-axis column

    return R_matrix
def rotate_axes(R_matrix, rotate_y=90, rotate_z=90):
    """Rotate the y (green) and z (blue) axes by specified angles."""
    # Create rotation matrices for rotating around x, y, and z
    rot_y = R.from_euler('y', rotate_y, degrees=True).as_matrix()  # Rotation around y-axis
    rot_z = R.from_euler('z', rotate_z, degrees=True).as_matrix()  # Rotation around z-axis

    # Apply the rotations to the existing rotation matrix
    R_new = np.dot(R_matrix, rot_z)  # Rotate z-axis first
    R_new = np.dot(R_new, rot_y)  # Rotate y-axis after

    return R_new

def visualize_trajectory_and_orientation(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    translations = []
    orientations = []

    # Extract translations and orientations from each pose
    for T in poses:
        t, euler_angles = extract_translation_rotation(T)
        translations.append(t)
        orientations.append(euler_angles)

    translations = np.array(translations)

    # Plot trajectory (translation vectors)
    ax.plot(translations[0, 0], translations[0, 1], translations[0, 2], label="Trajectory", color='blue', linewidth=2)



    #Visualize orientation at each pose with quivers
    for i, T in enumerate(poses):
        t, _ = extract_translation_rotation(T)
        R_matrix = T[:3, :3]

        # Draw the local coordinate axes at each pose
        if(i==0):
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='black', length=0.01,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='green', length=0.01,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='blue', length=0.01,
                      normalize=True)
            
        else:
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='red', length=0.01,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='green', length=0.01,
                      normalize=True)
            ax.quiver(t[0], t[1], t[2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='blue', length=0.01,
                      normalize=True)

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory and Orientation Visualization')

    plt.legend()
    plt.show()


visualize_trajectory_and_orientation(poses)
