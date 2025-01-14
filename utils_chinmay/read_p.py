import open3d as o3d
import numpy as np
import os
count = 1
def read_and_display_pcd_from_txt(txt_filename, num_points_to_display=5000, count=0):
    # Initialize lists for points, normals, and colors
    points = []
    normals = []
    colors = []

    # Read the text file and parse the data
    with open(txt_filename, 'r') as file:
        for line in file:
            # Skip empty lines or malformed lines
            if line.strip() == "":
                continue

            # Split the line into values
            values = line.split()

            # Ensure the line has at least 7 values (x, z, y, normal_x, normal_y, normal_z, label)
            if len(values) < 7:
                continue

            try:
                # Extract point (x, z, y), normal, and label
                x, z, y = float(values[0]), float(values[1]), float(values[2])
                normal = [float(values[3]), float(values[4]), float(values[5])]
                label = int(values[6])

                # Add the point, normal, and color (black if label is 1, grey if label is in the specified list)
                points.append([x, z, y])
                normals.append(normal)
                if label in [0, 2, 4, 6, 8, 10, 12, 14, 16]:
                    colors.append([0.8, 0.8, 0.8])  # Grey
                else:
                    colors.append([0, 0, 0])  # Black

            except ValueError:
                # Handle any lines with incorrect data (e.g., non-numeric values)
                continue

    # Convert the lists into numpy arrays
    points = np.array(points)
    normals = np.array(normals)
    colors = np.array(colors)
    string_toDisp = ''
    # Check if there are more points than num_points_to_display
    if len(points) > num_points_to_display:
        # Randomly sample the points, normals, and colors
        indices = np.random.choice(len(points), num_points_to_display, replace=False)
        points = points[indices]
        normals = normals[indices]
        colors = colors[indices]
        string_toDisp = 'above30k' 
        print(string_toDisp)

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #pcd.normals = o3d.utility.Vector3dVector(normals)  # Include normals
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name = str(count)+string_toDisp)
    

# Example usage:
txt_filename = '/home/student/points_and_colors.txt'  # Replace with your file name
read_and_display_pcd_from_txt(txt_filename, num_points_to_display=30000)
exit()
base_folder = "/home/chinmayn/part_object/"
num_points_to_display = 60000
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)

    # Exclude the meta_data folder and ensure the path is a directory
    if folder_name == "pinch_pull":
    

        # Iterate through txt files in the current folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):  # Check for txt files
                file_path = os.path.join(folder_path, file_name)
                print(file_path)
                read_and_display_pcd_from_txt(file_path, num_points_to_display,count)
                count += 1

