import os
import json
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from pathlib import Path

def load_json_indices(json_file_path):
    """Load indices from a JSON file."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    annotations = data['annotations']
    return {ann['annot_id']: { 'indices': ann['indices'], 'label': ann['label']} for ann in annotations}

def load_and_extract_ply_points(folder_path, indices_dict):
    """Load PLY files and extract points based on indices."""
    extracted_points = {}
    
    for root_dir, dirs, files in os.walk(folder_path):
        if os.path.basename(root_dir).startswith('4'):
            for file in files:
                if file.endswith('.ply'):
                    ply_path = os.path.join(root_dir, file)
                    ply = o3d.io.read_point_cloud(ply_path)
                    
                    # Get points as a numpy array
                    points = np.asarray(ply.points)
                    colors = np.asarray(ply.colors)
                    
                    # Extract points for each annotation ID and save in a dictionary
                    for annot_id, annot in indices_dict.items():
                        selected_points = points[annot['indices']]  # Extract points based on indices
                        selected_colors = colors[annot['indices']]
                        extracted_points[annot_id] = {}
                        extracted_points[annot_id]['ply_path'] = ply_path
                        extracted_points[annot_id]['points'] = selected_points
                        extracted_points[annot_id]['label'] = annot['label']
                        extracted_points[annot_id]['colors'] = selected_colors
                
    return extracted_points,ply

def save_point_cloud_as_off(pcd, output_file_path, output_filename):
    """
    Save a point cloud as an OFF file.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud to save.
        output_file_path (str): The file path to save the OFF file.
    """
    # Convert the points to a numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Open the file for writing
    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    with open(output_file_path+output_filename, 'w') as file:
        # Write the OFF header
        file.write("OFF\n")
        num_vertices = len(points)
        num_faces = 0  # Faces are not defined in point clouds
        file.write(f"{num_vertices} {num_faces} 0\n")
        
        # Write each point and optional color
        for i in range(num_vertices):
            if colors is not None:
                # Write point coordinates and RGB colors
                file.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
            else:
                # Write only point coordinates
                file.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")
    
    # f = open(output_file_path+"train.txt", "a")
    # f.write(output_file_path+output_filename+'\n')
    # f.close()
    print(f"Point cloud saved as OFF file at: {output_file_path}")

def save_points_as_ply_and_label_as_txt2(points, label, output_file_path, annot_id,scene_id):
    """
    Save points as a PLY file.
    
    Parameters:
        points (np.ndarray or o3d.utility.Vector3dVector): Points to save, as an Nx3 array or Open3D point cloud points.
        output_file_path (str): Path to save the PLY file.
    """
    # Create a new PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Check if points are already in Open3D format; otherwise, convert
    if not isinstance(points, o3d.utility.Vector3dVector):
        points = o3d.utility.Vector3dVector(points)
    
    pcd.points = points
    
    # Save the point cloud to a PLY file
    
    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    my_file = Path(output_file_path+'/'+annot_id+".ply")
    #print(annot_id,scene_id)
    if my_file.is_file():
        # file exists
        print(f"File: {output_file_path+'/'+annot_id+'.ply'}, already exists")
    else:
        o3d.io.write_point_cloud(output_file_path+'/'+annot_id+".ply", pcd)
        print(f"PLY file saved at: {output_file_path+'/'+annot_id+'.ply'}")
    f = open(output_file_path+'/'+"label.txt", "a")
    f.write(annot_id+','+label+'\n')
    f.close()

def label_background_around_handle(pcd, door_handle_pcd, radius=0.05):
    """Label a small region around the door handle as background."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Get the points of the door handle
    door_handle_points = np.asarray(door_handle_pcd.points)
    
    # Find the points in the cloud within a radius from the door handle points
    print(door_handle_pcd)
    #distances = np.linalg.norm(points[:, None] - door_handle_points, axis=2)
    
    points_np = points
    door_handle_points_np = door_handle_points

    # Build the tree for the door handle points
    tree = KDTree(door_handle_points_np)

    # Query distances and indices for all points in the point cloud
    distances, indices = tree.query(points_np, k=1)  # k=1 for the nearest neighbor
    print(distances)
    
    # Create a mask for points inside the small region (within radius from door handle)
    background_mask = np.any(distances <= radius, axis=1)
    print(background_mask)
    
    selected_points = points[background_mask]
    selected_colors = colors[background_mask]
    
    # Create a new point cloud with the selected points and colors
    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
    selected_pcd.colors = o3d.utility.Vector3dVector(selected_colors)
    
    return selected_pcd
    
    return pcd

def save_points_as_ply_and_label_as_txt(points, label, output_file_path):
    """
    Save points as a PLY file.
    
    Parameters:
        points (np.ndarray or o3d.utility.Vector3dVector): Points to save, as an Nx3 array or Open3D point cloud points.
        output_file_path (str): Path to save the PLY file.
    """
    # Create a new PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Check if points are already in Open3D format; otherwise, convert
    if not isinstance(points, o3d.utility.Vector3dVector):
        points = o3d.utility.Vector3dVector(points)
    
    pcd.points = points
    
    # Save the point cloud to a PLY file
    
    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    my_file = Path(output_file_path+"object.ply")
    if my_file.is_file():
        # file exists
        print(f"File: {output_file_path+'object.ply'}, already exists")
    else:
        o3d.io.write_point_cloud(output_file_path+'object.ply', pcd)
        print(f"PLY file saved at: {output_file_path+'object.ply'}")
    f = open(output_file_path+"label.txt", "w")
    f.write(label)
    f.close()

# Define paths
folder_path = "/home/student/dev/"  # Folder containing PLY files
json_file_path = "path/to/your/indices.json"  # JSON file with indices

labels_to_int = {
    'tip_push':0,
    'hook_turn':1,
    'exclude':2,
    'hook_pull':3,
    'key_press':4,
    'rotate':5,
    'foot_push':6,
    'unplug':7,
    'plug_in':8,
    'pinch_pull':9
}
# Load indices and extract points from PLY files
labels = {}
train_files = []
count = 0
for root, dirs, files in os.walk(folder_path):
        if len(dirs) > 2:
            for dir in dirs:
                if dir.startswith('4'):
                    json_file_path = root+dir+'/'+dir+'_annotations.json'
                    indices_dict = load_json_indices(json_file_path)
                    extracted_points,ply = load_and_extract_ply_points(root+dir, indices_dict)

                    # Example to print or use the extracted points
                    for annot_id, annot in extracted_points.items():
                        # print(f"File: {annot['ply_path']}, Annotation ID: {annot_id}")
                        # print(annot['points'])
                        # print(annot['label'])
                        file_path = annot['ply_path'].split('/')[0:-1]
                        new_ply_path = '/'.join(file_path)
                        # print(new_ply_path)
                        ply_path = '/home/student/train_without_exclude_class_off'
                        if(len(annot['points'])>=200 and annot['label']!='exclude'):
                            print(len(annot['points']))
                            #save_points_as_ply_and_label_as_txt2(annot['points'],str(labels_to_int[annot['label']]),ply_path,annot_id,json_file_path)
                            
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(annot['points'])  # Random points for testing
                            pcd.colors = o3d.utility.Vector3dVector(annot['colors'])  # Optional colors
                            
                            background = label_background_around_handle(ply,pcd)
                            # print(background)

                            #save_point_cloud_as_off(pcd,ply_path+'/',annot_id+'.off')
                            
                            train_files.append(ply_path+'/'+annot_id+'.off')
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(annot['points'])
                            pcd.colors = o3d.utility.Vector3dVector(annot['colors'])
                            #Visualize the point cloud
                            #o3d.visualization.draw_geometries([pcd,ply], window_name=annot['label'])
                            #exit()
                            aabb = pcd.get_oriented_bounding_box()
                            print(aabb)
                            # Display the bounding box with the point cloud
                            aabb.color = (1, 0, 0)  # Set color for visualization
                            min_bound = aabb.get_min_bound()  # Minimum x, y, z corner
                            max_bound = aabb.get_max_bound()  # Maximum x, y, z corner
                            # center = aabb.get_center()        # Center of the bounding box
                            # extent = aabb.get_extent()        # Width, height, depth
                            labels[annot_id] = str(labels_to_int[annot['label']])+','+str(min_bound)+','+str(max_bound)
                            o3d.visualization.draw_geometries([pcd,background , aabb], window_name=annot['label'])
                        count += 1
                        
sorted_data_dict = dict(sorted(labels.items()))

final_labels = [label for label in list(sorted_data_dict.values())]
sorted_files = sorted(train_files)
with open(ply_path+'/'+'train.txt', 'w') as file:
    for item in sorted_files:
        file.write(f"{item}\n")

with open(ply_path+'/'+'label.txt', 'w') as file:
    for item in final_labels:
        file.write(f"{item}\n")
# ply_files = sorted([f for f in os.listdir(ply_path) if f.endswith('.off')])

# import pandas as pd
# df = pd.DataFrame(list(sorted_data_dict.items()), columns=['Key', 'Value'])

# # Write the DataFrame to an Excel file
# df.to_excel('outputcc.xlsx', index=False)

# df2 = pd.DataFrame(data=ply_files, columns=['Column1'])

# # Write the DataFrame to an Excel file
# df2.to_excel('output1cc.xlsx', index=False)
