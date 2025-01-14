import json
import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
id_to_cat = {1: 'wall',
 2: 'chair',
 3: 'book',
 4: 'furniture',
 5: 'door',
 6: 'floor',
 7: 'unknown',
 8: 'table',
 50: 'window',
 29: 'screen',
 11: 'sitting bed',
 12: 'box',
 13: 'picture',
 14: 'ceiling',
 15: 'cloth',
 16: 'sink',
 17: 'bag',
 18: 'lamp',
 19: 'drawer',
 20: 'curtain',
 21: 'mirror',
 22: 'plant',
 91: 'heater',
 24: 'tissue paper',
 25: 'footwear',
 26: 'bottle',
 27: 'top',
 28: 'toilet',
 30: 'fridge',
 31: 'cup',
 32: 'phone',
 33: 'electronics',
 34: 'shower',
 35: 'fan',
 36: 'paper',
 37: 'bathroom',
 38: 'bar',
 39: 'switch',
 40: 'kitchen appliance',
 41: 'decoration',
 42: 'range hood',
 43: 'board',
 75: 'clock',
 44: 'railing',
 45: 'mat',
 46: 'person',
 47: 'stairs',
 48: 'dumbell',
 49: 'pillar',
 51: 'signboard',
 52: 'dishwasher',
 53: 'washing machine',
 54: 'piano',
 55: 'cart',
 56: 'blinds',
 57: 'dish rack',
 58: 'mail box',
 59: 'bicycle',
 60: 'ladder',
 61: 'rack',
 62: 'tray',
 63: 'paper cutter',
 64: 'plunger',
 65: 'guitar',
 66: 'fire extinguisher',
 67: 'pitcher',
 68: 'pipe',
 69: 'plate',
 70: 'bowl',
 71: 'closet rod',
 72: 'scale',
 73: 'broom',
 74: 'toy',
 76: 'ironing board',
 77: 'fire alarm',
 78: 'fireplace',
 79: 'vase',
 80: 'vent',
 81: 'candle',
 82: 'dustpan',
 83: 'jar',
 84: 'rod',
 85: 'step',
 86: 'step stool',
 87: 'vending machine',
 88: 'coat hanger',
 89: 'water fountain',
 90: 'basket',
 92: 'banner',
 93: 'iron ',
 94: 'soap',
 95: 'cutting board',
 96: 'kitchen island',
 97: 'sleeping bag',
 98: 'tire',
 99: 'toothbrush',
 100: 'faucet',
 101: 'thermos',
 102: 'tripod',
 103: 'dispenser',
 104: 'remote',
 105: 'stapler',
 106: 'treadmill',
 107: 'dart board',
 108: 'metronome',
 109: 'rope',
 110: 'object',
 111: 'water heater',
 112: 'hair brush',
 113: 'doll house',
 114: 'envelope',
 115: 'food',
 116: 'frying pan',
 117: 'helmet',
 118: 'tennis racket',
 119: 'umbrella'}
cat_id = {
    'tip_push':120,
    'hook_turn':121,
    'exclude':122,
    'hook_pull':123,
    'key_press':124,
    'rotate':125,
    'foot_push':126,
    'unplug':127,
    'plug_in':128,
    'pinch_pull':129    
}

def read_file_to_dict(file_path):
    """
    Reads a text file where each line contains two comma-separated values,
    and returns a dictionary with the first value as the key and the second as the value.

    :param file_path: Path to the text file
    :return: Dictionary with key-value pairs from the file
    """
    result_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip any extra whitespace and split the line by comma
                parts = line.strip().split(',')
                if len(parts) == 2:  # Ensure the line has exactly two parts
                    key, value = parts[0].strip(), parts[1].strip()
                    result_dict[key] = value
                else:
                    print(f"Skipping malformed line: {line.strip()}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return result_dict

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

def find_colors(points, scene_points, scene_colors):
    tree = KDTree(scene_points)
    _, indices = tree.query(points[:, :3])  # Match using x, y, z
    colors = scene_colors[indices]
    return colors

def load_text_file(file_path):
    data = np.loadtxt(file_path)
    return data

def get_train_test_val_dict(parent_path):
    with open(parent_path+'shuffled_val_file_list.json', 'r') as file:
        val_list = json.load(file)
    with open(parent_path+'shuffled_train_file_list.json', 'r') as file:    
        train_list = json.load(file)
    with open(parent_path+'shuffled_test_file_list.json', 'r') as file:
        test_list = json.load(file)

    val_return = []
    train_return = []
    test_return = []
    for id in val_list:
        val_return.append(id.split('/')[-1])
    for id in train_list:
        train_return.append(id.split('/')[-1])
    for id in test_list:
        test_return.append(id.split('/')[-1])

    print(len(val_return), len(test_return), len(train_return))

    return train_return, test_return, val_return

directory_name = {
    'tip_push':20,
    'hook_turn':21,
    'exclude':22,
    'hook_pull':23,
    'key_press':24,
    'rotate':25,
    'foot_push':26,
    'unplug':27,
    'plug_in':28,
    'pinch_pull':29    
}
annot_visit_map = {}
part_metadata = {}
train_dict, test_dict, val_dict = get_train_test_val_dict('/home/student/move/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/')
folder_path = '/home/student/dev'
text_files_home = '/home/student/move/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
count = 0
for root, dirs, files in os.walk(folder_path):
        if len(dirs) > 2:
            for dir in dirs:
                if dir.startswith('4'):
                    json_file_path = root+'/'+dir+'/'+dir+'_annotations.json'
                    with open(json_file_path, 'r') as file:
                         data = json.load(file)
                    
                    annotations = data['annotations']
                    for annotation in annotations:
                        if(annotation['label'] in directory_name):
                            if os.path.exists(text_files_home+'meta_data/'+annotation['label']+'/'+annotation['annot_id']+'.json'):
                                with open(text_files_home+'meta_data/'+annotation['label']+'/'+annotation['annot_id']+'.json') as file:
                                    metadata = json.load(file)
                                if (len(metadata['object_label']) > 2
                                    ):
                                    objects1 = ''
                                    for obj in metadata['object_label']:
                                        objects1 = objects1 + id_to_cat[obj] +'('+str(obj)+')'+ '_'
                                    #print(str(count + 1), objects1, annotation['annot_id'],annotation['label'])
                                    #print(text_files_home+str(directory_name[annotation['label']])+'/'+annotation['annot_id']+'.txt')
                                    count += 1
                                
                                part_metadata[text_files_home+str(directory_name[annotation['label']])+'/'+annotation['annot_id']+'.txt'] = metadata
                                annot_visit_map[text_files_home+str(directory_name[annotation['label']])+'/'+annotation['annot_id']+'.txt'] = root+'/'+dir+'/'+dir+'_laser_scan.ply'

morethan2 = read_file_to_dict('/home/student/moreThan2ObjectLabel.txt')
exact2 = read_file_to_dict('/home/student/2classes_scene_annot.txt')


write_to_txt = ''
count = 0
object_grey_labels = [0,2,4,6,8,10,12,14,16]
try:
    for annot_id,scene_path in annot_visit_map.items():
        if os.path.exists(annot_id):
            print('**************************{}*******************************'.format(count))
            print(scene_path)
            print(annot_id)
            print(part_metadata[annot_id])
            
            txt_pcd_points = load_text_file(annot_id)
            len_pcd = len(txt_pcd_points)
            array = np.zeros((len_pcd, 12))
            
            pcd = o3d.io.read_point_cloud(scene_path)
            scene_points, scene_colors = np.asarray(pcd.points), np.asarray(pcd.colors)
            txt_pcd_points[:, [1,2]] = txt_pcd_points[:, [2,1]]
            tx_pcd_colors = find_colors(txt_pcd_points, scene_points, scene_colors)

            count += 1
            len_pcd_count = 0
            for point in txt_pcd_points:
                if point[-1] in object_grey_labels:
                    segment = '0'
                    if(len( part_metadata[annot_id]['object_label']) == 2):
                        label = exact2[annot_id]
                    elif(len( part_metadata[annot_id]['object_label']) > 2):
                        label = morethan2[annot_id]
                    else:
                        label = str(part_metadata[annot_id]['object_label'][0])
                    instance_id = '0'
                else:
                    segment = '1'
                    label = str(cat_id[part_metadata[annot_id]['part_label']])
                    instance_id = '1'
                array[len_pcd_count, 0:3] = point[:3] # Random xyz values
                array[len_pcd_count, 3:6] = np.random.rand(1, 3)  # Random color values
                array[len_pcd_count, 6:9] = point[3:6]  # Random normal values
                array[len_pcd_count, 9] = segment  # Random segment IDs (0-9)
                array[len_pcd_count, 10] = label  # Random labels (0-4)
                array[len_pcd_count, 11] = instance_id  # Unique instance IDs
                len_pcd_count += 1
            
            array[:,3:6] = tx_pcd_colors
            an_id = annot_id.split('/')[-1]
            an_id_only = an_id.replace('.txt', '')
            if an_id in train_dict:
                np.save('/home/student/Mask3D_SIR/Train/'+an_id_only, array)
            elif an_id in test_dict:
                np.save('/home/student/Mask3D_SIR/Test/'+an_id_only, array)
            else:
                np.save('/home/student/Mask3D_SIR/Val/'+an_id_only, array)
            #save array
except Exception as e:
    print(e)
