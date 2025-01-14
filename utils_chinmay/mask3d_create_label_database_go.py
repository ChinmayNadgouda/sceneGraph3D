import yaml
import json
import numpy as np
import os

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
def calculate_color_stats_len(filepath):
    color_data = np.load(filepath)  # Assuming .npy file contains RGB values
    length = len(color_data)
    label = np.unique(color_data[:,-2])
    label = [ x for x in label if x <= 119 ]
    if len(label) == 0:
        label = np.unique(color_data[:,-2])
    features = color_data[:, 3:6]

    # Compute color mean and color std as per the new method
    filebase = {}
    filebase["color_mean"] = [
        float((features[:, 0] ).mean()),
        float((features[:, 1] ).mean()),
        float((features[:, 2] ).mean()),
    ]
    filebase["color_std"] = [
        float(((features[:, 0] ) ** 2).mean()),
        float(((features[:, 1] ) ** 2).mean()),
        float(((features[:, 2] ) ** 2).mean()),
    ]
    # color_mean = color_data[:,3:6].mean(axis=0).tolist()
    # color_std = color_data[:,3:6].std(axis=0).tolist()
    return filebase["color_mean"] , filebase["color_std"], length, label[0]
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
 119: 'umbrella',
 }
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

all_cats = {
    1: 'wall',
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
 119: 'umbrella',
    120:'tip_push',
    121:'hook_turn',
    122:'exclude',
    123:'hook_pull',
    124:'key_press',
    125:'rotate',
    126:'foot_push',
    127:'unplug',
    128:'plug_in',
    129:'pinch_pull'  
}

for key, value in all_cats.items():
    print('\''+str(value)+'\',')
exit()
colors = []
# Read YAML file
with open('/home/student/Mask3D/data/processed/scannet/label_database.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Process and print the contents
for key, value in data.items():
    colors.append(value['color'])

#label database
new_dict = {}
for key, value in all_cats.items():
    new_dict[key] = {}
    new_dict[key]['name'] = value
    new_dict[key]['color'] = colors[key]
    new_dict[key]['validation'] = True

yaml_s = yaml.dump(new_dict, default_flow_style=False)
with open('/home/gokul/Mask3D_SIR/label_database.yaml', 'w') as file:
        yaml.dump(new_dict, file, default_flow_style=False)

exit()
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
                                annot_visit_map[annotation['annot_id']] = dir
train_path = '/home/gokul/Mask3D_SIR/Train/'
train_path2 = './data/processed/scannet2/train/'
test_path = '/home/gokul/Mask3D_SIR/Test/'
val_path = '/home/gokul/Mask3D_SIR/Val/'
instance_path = '/home/gokul/Mask3D_SIR/instance_gt/'
instance_path2 = './data/processed/scannet2/instance_gt/train/'
def get_train_database(train_dict, annot_visit_map):
    count = 0
    yaml_data = []
    for train_id in train_dict:
        id = train_id.split('/')[-1].replace('.txt','')
        color_mean, color_std, length, label = calculate_color_stats_len(train_path+f"{id}.npy")
        scene = annot_visit_map[id]
        entry = {
            "color_mean": color_mean,
            "color_std": color_std,
            "file_len": length,  # Example length, replace with actual if needed
            "filepath": train_path2+f"{id}.npy",
            "instance_gt_filepath": instance_path2+f"train/{id}.txt",
            "raw_description_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}.txt",
            "raw_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean_2.ply",
            "raw_instance_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean.aggregation.json",
            "raw_label_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean_2.labels.ply",
            "raw_segmentation_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean_2.0.010000.segs.json",
            "scene": scene,
            "scene_type": int(label),
            "sub_scene": scene,
        }
        yaml_data.append(entry)
    with open('/home/gokul/Mask3D_SIR/train_database.yaml', 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
    with open('/home/gokul/Mask3D_SIR/train_validation_database.yaml', 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
def get_test_database(test_dict, annot_visit_map):
    count = 0
    yaml_data = []
    for train_id in test_dict:
        id = train_id.split('/')[-1].replace('.txt','')
        color_mean, color_std, length, label = calculate_color_stats_len(test_path+f"{id}.npy")
        scene = annot_visit_map[id]
        entry = {
            "file_len": length,  # Example length, replace with actual if needed
            "filepath": test_path+f"{id}.npy",
            "raw_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean_2.ply",
            "scene": scene,
            "sub_scene": scene,
        }
        yaml_data.append(entry)
    with open('/home/gokul/Mask3D_SIR/test_database.yaml', 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
def get_val_database(val_dict, annot_visit_map):
    count = 0
    yaml_data = []
    for train_id in val_dict:
        id = train_id.split('/')[-1].replace('.txt','')
        color_mean, color_std, length, label = calculate_color_stats_len(val_path+f"{id}.npy")
        scene = annot_visit_map[id]
        entry = {
            "color_mean": color_mean,
            "color_std": color_std,
            "file_len": length,  # Example length, replace with actual if needed
            "filepath": val_path+f"{id}.npy",
            "instance_gt_filepath": instance_path+f"validation/{id}.txt",
            "raw_description_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}.txt",
            "raw_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean_2.ply",
            "raw_instance_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean.aggregation.json",
            "raw_label_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean_2.labels.ply",
            "raw_segmentation_filepath": f"data/raw/scannet/scannet/scans/scene{id}/scene{id}_vh_clean_2.0.010000.segs.json",
            "scene": scene,
            "scene_type": int(label),
            "sub_scene": scene,
        }
        yaml_data.append(entry)
    with open('/home/gokul/Mask3D_SIR/validation_database.yaml', 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
    with open('/home/gokul/Mask3D_SIR/train_validation_database.yaml', 'a') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

def get_mean_std_colors(train_dict,annot_visit_map):
    count = 0
    yaml_data = []
    color_means = []
    color_stds = []
    for train_id in train_dict:
        id = train_id.split('/')[-1].replace('.txt','')
        color_mean, color_std, length, label = calculate_color_stats_len(train_path+f"{id}.npy")
        color_means.append(color_mean)
        color_stds.append(color_std)
    meann = np.array(color_means).mean(axis=0)
    #stdd = np.array(color_stds).std(axis=0) 
    stdd = np.sqrt(np.array(color_stds).mean(axis=0) -meann**2)
 
    yaml_data = [{'color_mean': [float(x) for x in meann], 'color_std': [float(x) for x in stdd]}] 
    print(yaml_data)
    with open('/home/gokul/Mask3D_SIR/color_mean_std.yaml', 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

def get_mean_std_colors_mask3d(path):
    yaml_data = []
    color_means = []
    color_stds = []
    for root, dirs, files in os.walk(path):
        print(files)
        for file in files:
            color_mean, color_std, length, label = calculate_color_stats_len(path+str(file))
            color_means.append(color_mean)
            color_stds.append(color_std)
        meann = np.array(color_means).mean(axis=0)
        stdd = np.array(color_stds).std(axis=0) 
        stdd = np.sqrt(np.array(color_stds).mean(axis=0) -meann**2)
    
        yaml_data = [{'color_mean': [float(x) for x in meann], 'color_std': [float(x) for x in stdd]}] 
        print(yaml_data)
def get_instance_gt(train_dict, val_dict, annot_visit_map):
    for train_id in train_dict:
        id = train_id.split('/')[-1].replace('.txt','')
        data = np.load(train_path+f"{id}.npy")
        labels = data[:, -2]
        instance_id = data[:,-1]

        gt = labels * 1000 + instance_id + 1
        np.savetxt(instance_path+'train/'+id+'txt', gt.astype(np.int32), fmt="%d")
    for val_id in val_dict:
        id = val_id.split('/')[-1].replace('.txt','')
        data = np.load(val_path+f"{id}.npy")
        labels = data[:, -2]
        instance_id = data[:,-1]

        gt = labels * 1000 + instance_id + 1
        np.savetxt(instance_path+'validation/'+id+'txt', gt.astype(np.int32), fmt="%d")
        
get_instance_gt(train_dict, test_dict, annot_visit_map)
exit()
get_mean_std_colors(train_dict, annot_visit_map)
#mask3d_path = '/home/student/Mask3D/data/222/scannet/train/'
#get_mean_std_colors_mask3d(mask3d_path)
#exit()
get_train_database(train_dict, annot_visit_map)
get_test_database(test_dict, annot_visit_map)
get_val_database(val_dict, annot_visit_map)
