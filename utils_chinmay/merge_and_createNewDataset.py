import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import json
import os
import matplotlib.pyplot as plt
import time
from itertools import islice
#import torch
from collections import defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def get_colors_for_id(number_of_colors):
    file_path = "label_mapping.csv"
    dataframe = pd.read_csv(file_path)
    id_to_color = [[0.1, 0.8, 0.5]]
    for i in range(1,number_of_colors+1):
        color = dataframe.loc[dataframe['wn199'] == float(i), 'color'].iloc[0]
        rgb_values = list(map(int, color.split('-')))
        normalized_rgb = [val / 255.0 for val in rgb_values]
        clipped_rgb = np.clip(normalized_rgb, 0, 1).tolist()
        id_to_color.append(clipped_rgb)
    
    return id_to_color


label_to_cc = "{1: [1], 2: [2, 18, 41, 50, 60, 75, 170], 3: [3], 4: [4, 10, 61, 129, 131, 181], 5: [5, 16], 6: [6], 7: [7], 8: [8, 17, 37, 59, 166], 50: [9, 85], 29: [11, 39, 42, 52, 78, 137], 11: [12, 20, 26, 27, 62], 12: [13, 77, 81, 84, 135, 177], 13: [14, 153, 173], 14: [15], 15: [19, 25, 46, 114, 144, 155, 159], 16: [21, 47], 17: [22, 34, 51, 91, 93, 98], 18: [23, 45], 19: [24], 20: [28, 48], 21: [29], 22: [30], 91: [31, 148, 165], 24: [32, 90], 25: [33, 161], 26: [35], 27: [36, 56], 28: [38], 30: [40], 31: [43], 32: [44], 33: [49, 53, 64, 88, 106, 112, 121, 128, 139, 175], 34: [54, 55, 66], 35: [57], 36: [58], 37: [63], 38: [65], 39: [67], 40: [68, 103, 116, 117, 118], 41: [69], 42: [70], 43: [71], 75: [72, 124], 44: [73, 76], 45: [74, 125], 46: [79], 47: [80], 48: [82], 49: [83], 51: [86], 52: [87], 53: [89], 54: [92], 55: [94], 56: [95], 57: [96], 58: [97], 59: [99], 60: [100], 61: [101, 122], 62: [102], 63: [104], 64: [105], 65: [107], 66: [108], 67: [109], 68: [110], 69: [111], 70: [113], 71: [115], 72: [119], 73: [120], 74: [123], 76: [126], 77: [127], 78: [130], 79: [132], 80: [133], 81: [134], 82: [136], 83: [138], 84: [140], 85: [141], 86: [142], 87: [143], 88: [145], 89: [146], 90: [147], 92: [149], 93: [150], 94: [151], 95: [152], 96: [154], 97: [156], 98: [157], 99: [158], 100: [160], 101: [162], 102: [163], 103: [164], 104: [167, 180], 105: [168], 106: [169], 107: [171], 108: [172], 109: [174], 110: [176, 0], 111: [178], 112: [179], 113: [182], 114: [183], 115: [184], 116: [185], 117: [186], 118: [187], 119: [188]}"
# Generate 116 distinct colors using a colormap
def generate_colors(num_colors=120):
    cmap = plt.cm.get_cmap("hsv", num_colors)  # "viridis" is perceptually uniform
    colors = cmap(np.linspace(0, 1, num_colors))[:, :3]  # Extract RGB values (exclude alpha)
    return colors

# Get the color corresponding to a number
def get_color_for_number(number, colors):
    if 1 <= number <= len(colors):
        return colors[number - 1]
    else:
        raise ValueError("Number is out of range!")
def get_interestPoints_from_parts_indices(pcd, annotations_path):
    """Load indices from a JSON file."""
    points = np.asarray(pcd.points)
    with open(annotations_path, 'r') as file:
        data = json.load(file)
    annotations = data['annotations']
    indices = []
    excluded_indices = []
    labelss = {}
    only_label = []
    label_to_points = {}
    for ann in annotations:
        if ann['label']!= 'exclude':
            for indice in ann['indices']:
                indices.append([int(indice), ann['label']])
                only_label.append([ann['label'], ann['annot_id']])
            label_to_points[ann['annot_id']] = {'label': ann['label'], 'indices': ann['indices'], 'points': []}
        else:
            for indice in ann['indices']:
                excluded_indices.append(indice)
    indices = np.asarray(indices)
    points_to_return = points[ [ int(x) for x in indices[:,0].tolist()] ]
    i = 0
    for point in points_to_return:
        labelss[tuple(point)] = only_label[i]
        label_to_points[only_label[i][1]]['points'].append(point.tolist())
        i += 1 
    return indices, points_to_return, labelss, label_to_points

def load_json_indices(json_file_path):
    """Load indices from a JSON file."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    annotations = data['annotations']
    indices = []
    excluded_indices = []
    for ann in annotations:
        if ann['label']!= 'exclude':
            for indice in ann['indices']:
                indices.append(indice)
        else:
            for indice in ann['indices']:
                excluded_indices.append(indice)
    return indices, excluded_indices


# Simulate cluster generation
def generate_clusters(points, colors, eps=0.05, min_points=100):
    color_groups = defaultdict(list)

    # Group points by unique color
    for point, color in zip(points, colors):
 
        color_to_compare = np.asarray([0.1, 0.8, 0.5]).astype(np.float64)
        if(color[0] == color_to_compare[0] and color[1] ==  color_to_compare[1] and color[2] == color_to_compare[2]):
            continue
        
        color_tuple = tuple(color)  # Use float values directly
        color_groups[color_tuple].append(point)

    # Process each color group
    clusters_by_color = defaultdict(list)
    for color, group_points in color_groups.items():
        group_points = np.asarray(group_points)  # Convert to NumPy array
        group_pcd = o3d.geometry.PointCloud()
        group_pcd.points = o3d.utility.Vector3dVector(group_points)
        group_pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(group_points), 1)))

        # Apply DBSCAN clustering
        labels = np.array(
            group_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
        )

        if labels.max() < 0:  # No clusters found
            continue

        # Store clusters as numpy arrays
        unique_labels = set(labels)
        for cluster_label in unique_labels:
            if cluster_label == -1:  # Ignore noise points
                continue
            cluster_points = group_points[labels == cluster_label]
            clusters_by_color[color].append(cluster_points)

    return clusters_by_color


# Subset matching and print original clusters
def find_matching_clusters(id_to_label_and_points, clusters_by_color):
    colorss = get_colors_for_id(120)
    all_clusters = [
         {"color": color, "points": cluster} for color, clusters in clusters_by_color.items() for cluster in clusters
    ]  # Flatten all clusters into one list
    part_object_byId = {}
    for id, value in id_to_label_and_points.items():
        points_set = set(map(tuple, value['points']))  # Convert points to a set

        # Check subset relationship with all clusters
        matching_clusters = [
            cluster for cluster in all_clusters
            if points_set & set(map(tuple, cluster["points"]))#if points_set.issubset(set(map(tuple, cluster)))  # Check subset
        ]

        # Output results
        if matching_clusters:
            all_cluster_points = [] 
            cluster_class = []
            #print(f"\nMatching clusters for ID {id} (label: {value['label']}):")
            for idx, cluster in enumerate(matching_clusters):
                #print(f"Cluster {idx + 1}: {len(cluster['points'])} points")
                all_cluster_points.append(np.array(cluster["points"]))
                true_color =  [float(x) for x in cluster["color"]]
                cluster_class.append(colorss.index(true_color)+1) 
            
            merged_points = np.vstack(all_cluster_points)
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
            # Create a color array for the merged_pcd (light grey)
            merged_colors = np.full((merged_points.shape[0], 3), [0.8, 0.8, 0.8], dtype=np.float64)  # Light grey
            merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            # print(merged_colors.shape, merged_pcd)
            # print(f"Shape of points: {np.asarray(value['points']).shape}")
            part_pcd = o3d.geometry.PointCloud()
            part_pcd.points = o3d.utility.Vector3dVector(np.asarray(value['points']).astype(np.float64))
            part_colors = np.full((np.asarray(value['points']).shape[0], 3), [0, 0, 0], dtype=np.float64)  # Black
            # print(part_colors.shape)
            part_pcd.colors = o3d.utility.Vector3dVector(part_colors)
            # print(part_pcd)

            # Find overlapping points using set intersection
            merged_points_set = set(map(tuple, merged_points))
            part_points_set = points_set
            overlapping_points = np.array(list(merged_points_set & part_points_set))  # Intersection

            # Find indices of overlapping points in merged_pcd
            if overlapping_points.size > 0:
                mask = np.isin(merged_points, overlapping_points).all(axis=1)
                merged_colors[mask] = [0.0, 0.0, 0.0]  # Black
                merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            part_object_byId[id] = {'pcds': [part_pcd, merged_pcd], 'meta_data': {'part_label': value['label'], 'object_label': cluster_class}}
            # print(part_object_byId)
            #o3d.visualization.draw_geometries([part_pcd, merged_pcd], window_name=str(value['label']))
        else:
            print(f"No matching clusters found for ID {id}")
    
    return part_object_byId
        


def get_parts_object_pcd(pcd, id_to_label_and_points):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    clusters_by_color = generate_clusters(points, colors)
    return find_matching_clusters(id_to_label_and_points, clusters_by_color)

def process_batch_with_colors(batch_indices, accurate_points, segmented_points, labels, 
                               segmented_tree, common_cat_to_label_map, colorss):
    batch_points = accurate_points[batch_indices]

    # Find nearest neighbors
    distances, indices = segmented_tree.query(batch_points, k=1)
    
    # Assign colors based on labels
    batch_colors = []
    for idx in indices:
        category_label = common_cat_to_label_map[int(labels[idx])]
        color = get_color_for_number(category_label, colorss)
        batch_colors.append(np.asarray(color).astype(np.float64))
    
    return np.array(batch_colors)

def process_point_clouds_parallel_with_colors(accurate_pcd, segmented_pcd, batch_size, labels, common_cat_to_label_map, num_workers=4):
    # Convert Open3D point clouds to NumPy arrays
    accurate_points = np.asarray(accurate_pcd.points)
    segmented_points = np.asarray(segmented_pcd.points)
    segmented_labels = np.asarray(segmented_pcd.colors)

    # Create KD-Tree for the segmented point cloud
    segmented_tree = cKDTree(segmented_points)

    # Prepare color palette
    colorss = get_colors_for_id(120)

    # Prepare batch indices
    total_points = accurate_points.shape[0]
    batches = [(i, min(i + batch_size, total_points)) for i in range(0, total_points, batch_size)]

    # Output array for accurate point colors
    accurate_labels = np.zeros((total_points, 3))  # Assuming RGB labels

    # Parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            lambda b: process_batch_with_colors(
                range(b[0], b[1]), 
                accurate_points, 
                segmented_points, 
                labels, 
                segmented_tree, 
                common_cat_to_label_map, 
                colorss
            ),
            batches
        ))
    # Combine results
    for i, (start, end) in enumerate(batches):
        accurate_labels[start:end] = results[i]

    # Add the labels as a property to the accurate point cloud
    accurate_pcd.colors = o3d.utility.Vector3dVector(accurate_labels)

    return accurate_pcd

def get_transformed_points(transformation_matrix, points):
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))


    # Apply the transformation matrix
    transformed_points = homogeneous_points @ transformation_matrix.T  # (N, 4)


    # Convert back to Cartesian coordinates (drop the homogeneous column)
    transformed_points_cartesian = transformed_points[:, :3]


    # Check if each point in array1 exists in array2
    transformed_points_cartesian = transformed_points_cartesian.astype(np.float64)

    return transformed_points_cartesian
def get_visitID_to_videoID_dict():
    # Load the Excel file
    file_path = 'union_of_scenes.xlsx'
    df = pd.read_excel(file_path)

    # Filter rows where T_F is 'Yes'
    filtered_df = df[df['T_F'] == 'Yes']

    # Group by 'visit_id' and create the dictionary with 'video_id' as the values
    result_dict = filtered_df.groupby('visit_id')['video_id'].apply(list).to_dict()

    # Display the resulting dictionary
    return result_dict

dict__ = get_visitID_to_videoID_dict()
scenefun_home = '/home/data/FunGraph/data/scenefun/'
labelmaker_home = '/home/chinmayn/arkitscenes/Training/'
visit_files = ['_crop_mask.npy', '_laser_scan.ply', '_annotations.json']
video_files = ['point_lifted_mesh.ply', 'labels.txt']
visit_video_files = '_transform.npy'
save_path = '/home/chinmayn/part_object/'
scenfun_label_to_pointnet_labels = {'tip_push':	[0,1],
'hook_turn': [2,3],
'hook_pull': [4,5],
'key_press': [6,7],
'rotate': [8,9],
'foot_push': [10,11],
'unplug': [12,13],
'plug_in':[14,15],
'pinch_pull': [16,17]}
for visit,video_list in dict__.items():
    print('started visit id: {}'.format(str(visit)))
    file_path = Path(scenefun_home+str(int(visit))+'/'+str(int(visit))+visit_files[1])  #laser scan
    if not file_path.exists():
        print(scenefun_home+str(int(visit))+'/'+str(int(visit))+visit_files[1])
        print("Exitting loop as the file mentioned above doesn't exist")
        continue
    pcd2 = o3d.io.read_point_cloud(str(file_path))

    points = np.asarray(pcd2.points)
    colors = np.asarray(pcd2.colors)

    mask = np.load(scenefun_home+str(int(visit))+'/'+str(int(visit))+visit_files[0])

    # Apply the mask (retain points where mask == 1)
    cropped_points = points
    cropped_colors = colors
    annotations_path = scenefun_home+str(int(visit))+'/'+str(int(visit))+visit_files[2]
    parts, excluded_parts = load_json_indices(annotations_path)
    part_object_byID_byVideo= {}
    for video in video_list:
        #start the computation and store files
        file_path2 = Path(scenefun_home+str(int(visit))+'/'+str(int(video))+'/'+str(int(video))+visit_video_files) #transform
        if not file_path2.exists():
            print(scenefun_home+str(int(visit))+'/'+str(int(video))+'/'+str(int(video))+visit_video_files)
            print("Exitting loop as the file mentioned above doesn't exist")
            continue
        transformation_matrix = np.load(str(file_path2))


        label_to_common_cat = pd.read_excel("wn199_to_common_cat_map.xlsx", engine='openpyxl')  # Load Excel file into a DataFrame

        # Convert to NumPy array
        label_to_common_cat_np = label_to_common_cat.to_numpy()

        common_cat_to_label_map = {}
        for index, row in enumerate(label_to_common_cat_np):
            common_cat_to_label_map[row[0]] = row[3]

        # Convert points to homogeneous coordinates
        # Shape: (N, 3) -> (N, 4) by adding a fourth column of ones
        num_points = cropped_points.shape[0]
        homogeneous_points = np.hstack((cropped_points, np.ones((num_points, 1))))


        # Apply the transformation matrix
        transformed_points = homogeneous_points @ transformation_matrix.T  # (N, 4)


        # Convert back to Cartesian coordinates (drop the homogeneous column)
        transformed_points_cartesian = transformed_points[:, :3]


        # Check if each point in array1 exists in array2
        transformed_points_cartesian = transformed_points_cartesian.astype(np.float64)
        if not np.all(np.isfinite(transformed_points_cartesian)):
            print("Found NaN or inf values in transformed points!")
            print("Invalid points:", transformed_points_cartesian[~np.isfinite(transformed_points_cartesian)])
            raise ValueError("Transformed points contain invalid values.")
        # Create a new point cloud with transformed points


        try:
            transformed_point_cloud = o3d.geometry.PointCloud()
            transformed_point_cloud.points = o3d.utility.Vector3dVector(transformed_points_cartesian)
            transformed_point_cloud.colors = o3d.utility.Vector3dVector(cropped_colors)
        except Exception as e:
            print(e)


        # Load point clouds
        accurate_pcd = transformed_point_cloud
        file_path3 = Path(labelmaker_home+str(int(video))+'/'+video_files[0])
        if not file_path3.exists():
            print(labelmaker_home+str(int(video))+'/'+video_files[0])
            print("Exitting loop as the file mentioned above doesn't exist")
            continue
        lifted_pcd = o3d.io.read_point_cloud(str(file_path3))
        lifted_pcd_labels = np.loadtxt(labelmaker_home+str(int(video))+'/'+video_files[1])
        #o3d.visualization.draw_geometries([lifted_pcd], window_name=str('Label Maker Arkit Point Lifted Mesh'))

        #o3d.visualization.draw_geometries([lifted_pcd, accurate_pcd], window_name=str('Overlapping LM and SF3D'))

        
        points_of_interest = []
        indices_labels, points_of_interest, label_dict, id_to_label_and_points = get_interestPoints_from_parts_indices(accurate_pcd, annotations_path)

        segmented_pcd = lifted_pcd
        # Process point clo uds
        batch_size = 100  # Adjust based on your resources

        points_cropped = np.asarray(accurate_pcd.points)[mask]
        colors_cropped = np.asarray(accurate_pcd.colors)[mask]

        accurate_pcd.points = o3d.utility.Vector3dVector(points_cropped)
        accurate_pcd.colors = o3d.utility.Vector3dVector(colors_cropped)
        
        #o3d.visualization.draw_geometries([accurate_pcd], window_name=str('Overlapping LM and SF3D'))
        # Record the start time
        print('started run for video id: {}'.format(str(video)))
        start_time = time.time()
        processed_pcd_new_colors = process_point_clouds_parallel_with_colors(accurate_pcd, segmented_pcd, batch_size, lifted_pcd_labels, common_cat_to_label_map, 8)
        #processed_point_cloud_parallel = process_point_clouds_parallel(accurate_pcd, segmented_pcd, batch_size, 8)
        # end_time = time.time()
        # # Calculate the elapsed time
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time: {elapsed_time} seconds")
        # #o3d.visualization.draw_geometries([processed_pcd_new_colors], window_name=str('segmented SceneFun3D'))

        # #processed_pcd = process_point_clouds(accurate_pcd, segmented_pcd, batch_size)

        # print('started2')
        # start_time = time.time()
        #get_segments_by_pcd_color__get_parts(processed_point_cloud_parallel,points_of_interest, label_dict)
        part_object_byID_byVideo[video] = get_parts_object_pcd(processed_pcd_new_colors, id_to_label_and_points)
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        # exit()
        # get_sorted_segments_by_label_colors_parts(lifted_pcd, lifted_pcd_labels, points_of_interest)
        # exit()

        # processed_pcd = accurate_pcd
        # processed_pcd_colors = np.asarray(processed_pcd.colors)
        # processed_pcd_colors[parts] = [0, 0, 0]
        # processed_pcd.colors = o3d.utility.Vector3dVector(processed_pcd_colors)

        # # Save the processed point cloud
        # o3d.io.write_point_cloud("processed_accurate_withparts.ply", processed_pcd)
        # o3d.visualization.draw_geometries([processed_pcd], window_name=str('segments'))

    id_part_obj = {}
    inverseTransformation_s_byId = {}
    for videoId, obj in part_object_byID_byVideo.items():
        transformation_matrix = np.load(Path(scenefun_home+str(int(visit))+'/'+str(int(videoId))+'/'+str(int(videoId))+visit_video_files)) #transform)
        inverse_transformation = np.linalg.inv(transformation_matrix)
        for id, obj2 in obj.items():
            pcds = obj2['pcds']
            meta_data = obj2['meta_data']
            if id in id_part_obj:
                # if transformation_id != key:
                pcds[0].points = o3d.utility.Vector3dVector(get_transformed_points(inverse_transformation, np.asarray(pcds[0].points).astype(np.float64)))
                pcds[1].points = o3d.utility.Vector3dVector(get_transformed_points(inverse_transformation, np.asarray(pcds[1].points).astype(np.float64)))
                id_part_obj[id]['pcds'].append(pcds[0])
                id_part_obj[id]['pcds'].append(pcds[1])
            else:
                # if transformation_id != key:
                #     pcds[0].points = o3d.utility.Vector3dVector(get_transformed_points(transformation_matrix, np.asarray(pcds[0].points).astype(np.float64)))
                #     pcds[1].points = o3d.utility.Vector3dVector(get_transformed_points(transformation_matrix, np.asarray(pcds[1].points).astype(np.float64)))
                pcds[0].points = o3d.utility.Vector3dVector(get_transformed_points(inverse_transformation, np.asarray(pcds[0].points).astype(np.float64)))
                pcds[1].points = o3d.utility.Vector3dVector(get_transformed_points(inverse_transformation, np.asarray(pcds[1].points).astype(np.float64)))
                id_part_obj[id] = {'pcds':[], 'meta_data': None}
                id_part_obj[id]['pcds'].append(pcds[0])
                id_part_obj[id]['pcds'].append(pcds[1])
            id_part_obj[id]['meta_data'] = meta_data

    for id, obj in id_part_obj.items():
        print('Saving pcd for id:{} with part label:{} and object label:{}'.format(str(id), str(obj['meta_data']['part_label']), str(obj['meta_data']['object_label'])))
        #o3d.visualization.draw_geometries(obj['pcds'], window_name=str(id))
        meta_data_output_folder_path  = save_path+'meta_data/'+str(obj['meta_data']['part_label'])+'/'
        output_folder_path = save_path+str(obj['meta_data']['part_label'])+'/'
        if not os.path.exists(output_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(output_folder_path)
        if not os.path.exists(meta_data_output_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(meta_data_output_folder_path)
        output_filename = output_folder_path+str(id)+'.txt'
        meta_data_filename = meta_data_output_folder_path+str(id)+'.txt'
        with open(meta_data_filename, 'w') as json_file:
                json.dump(obj['meta_data'], json_file, indent=4)
        if os.path.exists(output_filename):
        	print('Already exists')
        label_list = scenfun_label_to_pointnet_labels[str(obj['meta_data']['part_label'])]
        with open(output_filename, 'w') as file:
               for pcd in obj['pcds']:
		    # Compute normals
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
		    
		    # Get points and normals
                    points = np.asarray(pcd.points)
                    normals = np.asarray(pcd.normals)
                    colors = np.asarray(pcd.colors)
		    
		    # Iterate through the points and write to file
                    for i in range(len(points)):
                        x, y, z = points[i]
                        normal = normals[i]
                        color = colors[i]
                        # Check if the point is black (RGB = 0, 0, 0)
                        label = label_list[1] if np.all(color == 0) else label_list[0]
                        # Format the data: x z y normal_value1 normal_value2 normal_value3 label
                        line = f"{x} {z} {y} {normal[0]} {normal[1]} {normal[2]} {label}\n"
                        file.write(line)
        


