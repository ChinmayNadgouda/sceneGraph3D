import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import open3d as o3d
from open3d import pipelines
import numpy as np

# Initialize cv_bridge
bridge = CvBridge()

# Path to your ROS bag file
bag_file = '20240924_101254.bag'

# Output directories
rgb_dir = 'final_op_small_int/rgb/'
depth_dir = 'final_op_small_int/depth/'
pose_file = 'final_op_small_int/poses.txt'
pose_file_hybrid = 'final_op_small_int/poses_h.txt'

# Create directories if they don't exist
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

intrinsic_matrix_temp = np.array([[910.557, 0.0, 650.265], 
                            [0.0, 910.094, 358.224], 
                            [0.0, 0.0, 1.0]])  # Example for PrimeSense camera


# Function to estimate pose between two RGB-D frames
def estimate_odometry(source_color, source_depth, target_color, target_depth, prev_odo):
    # examples/Python/Basic/rgbd_odometry.py
    # Define camera intrinsic parameters
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()

    pinhole_camera_intrinsic.set_intrinsics(width=848, height=480, fx=214.313, fy=214.313, cx=214.517, cy=118.207)
    #pinhole_camera_intrinsic.set_intrinsics(width=1280, height=720, fx=910.557, fy=910.094, cx=650.265, cy=358.224)

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd_image, pinhole_camera_intrinsic)

    option = pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    
    if(prev_odo is not None):
        odo_init = prev_odo

    
    #print(option)

    [success_color_term, trans_color_term,
     info] = pipelines.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
         odo_init, pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term,
     info] = pipelines.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
         odo_init, pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    if success_color_term:
        #print("Using RGB-D Odometry")
        #print(trans_color_term)
        source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_color_term.transform(trans_color_term)
        #o3d.visualization.draw_geometries([target_pcd, source_pcd_color_term])
        
    if success_hybrid_term:
        #print("Using Hybrid RGB-D Odometry")
        #print(trans_hybrid_term)
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        #o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term])
    
    return trans_color_term, trans_hybrid_term

def plot_camera_trajectory2(poses, scale=0.01):
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
with open(pose_file, 'w') as pf, open(pose_file_hybrid, 'w') as pf_h:

    # Variables to store the previous frame
    prev_rgb = None
    prev_depth = None
    prev_timestamp = None
    camera_pose = np.eye(4)
    world_poses = [camera_pose]

    # Open the bag file
    with rosbag.Bag(bag_file, 'r') as bag:
        count = 0
        pose = None
        for topic, msg, t in bag.read_messages():
            timestamp = str(t.to_nsec())
            
            # Extract RGB images
            
            if topic == '/device_0/sensor_0/Depth_0/image/data':  # Adjust topic name if different
                depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                #depth_filename = os.path.join(depth_dir, f'depth_{count}.png')
                

            # Extract Depth images
            elif topic == '/device_0/sensor_1/Color_0/image/data':  # Adjust topic name if different
                rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                rgb_filename = os.path.join(rgb_dir, f'rgb_{count}.png')
                rgb_height, rgb_width = rgb_image.shape[:2]
                # Resize the depth image to match the RGB image dimensions
                resized_depth_image = cv2.resize(depth_image, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
                resized_depth_filename = os.path.join(depth_dir, f'depth_{count}_resized.png')
                cv2.imwrite(resized_depth_filename, resized_depth_image)
                cv2.imwrite(rgb_filename, rgb_image)

                
                # If previous frame exists, estimate pose
                if prev_rgb is not None and prev_depth is not None:
                    # Estimate the transformation between the current and previous frames
                    source_color = o3d.io.read_image(prev_rgb_filename)
                    source_depth = o3d.io.read_image(prev_resized_depth_filename)
                    target_color = o3d.io.read_image(rgb_filename)
                    target_depth = o3d.io.read_image(resized_depth_filename)
                    img1 = cv2.imread(prev_rgb_filename, cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(rgb_filename, cv2.IMREAD_GRAYSCALE)
                    pose, pose_h = estimate_odometry(source_color, source_depth, target_color, target_depth, pose)
                    if pose is not None:
                        world_poses.append(pose)
                    #print(pose)

                    if pose is not None:
                        temp_pose = ''
                        for poses in pose:
                            temp_pose = temp_pose + ' ' +' '.join(map(str, poses)).strip() 
                        # Save the pose (4x4 matrix) with the timestamp
                        temp_pose_h = ''
                        for poses in pose_h:
                            temp_pose_h = temp_pose_h + ' ' +' '.join(map(str, poses)).strip() 
                        pf.write(temp_pose.strip() +'\n')
                        pf_h.write(temp_pose_h.strip() +'\n')
                    else:
                        print(f"Odometry failed between frames {prev_timestamp} and {timestamp}.")

                # Update previous frame information
                prev_rgb_filename = rgb_filename
                #prev_depth_filename = depth_filename
                prev_resized_depth_filename = resized_depth_filename
                prev_rgb = rgb_image
                prev_depth = depth_image
                prev_timestamp = timestamp
                prev_count = count
                print(count)
                count += 1
                
        plot_camera_trajectory(world_poses)
        
            


# import pyrealsense2  as rs
# fname = '20240924_101254.bag'
# frame_present = True
# frameset = []
# count = 0
# d = rs.decimation_filter()
# cfg = rs.config()
# cfg.enable_device_from_file(fname, repeat_playback=False)
# # setup pipeline for the bag file
# pipe = rs.pipeline()
# # start streaming from file
# profile = pipe.start(cfg)
# while frame_present:
#     try:
#         frame = pipe.wait_for_frames()
#         frame.keep()
#         frameset.append(frame)
#         count += 1
#         if count == 30:
#             break
#     except RuntimeError:
#         print("number of frames extracted:", str(count))
#         frame_present = False
#         continue
# pipe.stop()

# # set alignment
# alignedset = []
# align_to = rs.stream.depth
# for f in frameset:
#     align = rs.align(align_to)
#     aligned = align.process(f)
#     depth = f.get_depth_frame()
#     image = f.get_color_frame()

#     processed = d.process(image)
#     prof = processed.get_profile().as_video_stream_profile()
#     intr = prof.get_intrinsics()
#     print(intr)


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
#         pf.write(temp_pose+'\n')