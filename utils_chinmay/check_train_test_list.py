import os
import json
home = '/home/student/dev/'
folder_path = home
visit_files = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
def get_interestPoints_from_parts_indices(visit_files, all_ids):
    """Load indices from a JSON file."""
    ann_id_to_visit_id = {}
    count = 0
    train_list = []
    val_list = []
    test_list = []
    for visit in visit_files:
        count += 1  
        annotations_path = home +str(int(visit))+'/'+ str(int(visit)) + '_annotations.json'
        if count < 160:
            with open(annotations_path, 'r') as file:
                data = json.load(file)
                annotations = data['annotations']
                for ann in annotations:
                     if ann['label'] != 'exclude':
                          if(ann['annot_id'] in all_ids):
                            train_list.append('/home/chinmayn/old_files/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/' + ann['annot_id'] +'.txt')
        elif count >= 160 and count < 200 :
             with open(annotations_path, 'r') as file:
                data = json.load(file)
                annotations = data['annotations']
                for ann in annotations:
                     if ann['label'] != 'exclude':
                         if(ann['annot_id'] in all_ids):
                            val_list.append('/home/chinmayn/old_files/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/' + ann['annot_id'] +'.txt')	        	
        else :
            with open(annotations_path, 'r') as file:
                data = json.load(file)
                annotations = data['annotations']
                for ann in annotations:
                    if ann['label'] != 'exclude':
                        if(ann['annot_id'] in all_ids):
                            test_list.append('/home/chinmayn/old_files/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/' + ann['annot_id'] +'.txt')

    return train_list, val_list, test_list


def match_dataset_to_visit_id(test_id, train_id_val_id, ann_id_to_visit_id):
    train_val_id = train_id_val_id
    train_val_visit_scenes = set()
    for id in train_val_id:
        train_val_visit_scenes.add(ann_id_to_visit_id[id])
    
    new_test_ids = []
    for id in test_id:
        if ann_id_to_visit_id[id] in train_val_visit_scenes:
            continue
        else:
            new_test_ids.append(id+'.txt')
    
    return new_test_ids

def get_train_test_ids(file_path):
    train_val_ids = []
    test_ids = []
    with open(file_path+'shuffled_train_file_list.json') as file:
        data = json.load(file)
        for file in data:
            id = file.split('/')[-1].replace('.txt','')
            train_val_ids.append(id)\
    
    with open(file_path+'shuffled_val_file_list.json') as file:
        data = json.load(file)
        for file in data:
            id = file.split('/')[-1].replace('.txt','')
            train_val_ids.append(id)

    with open(file_path+'shuffled_test_file_list.json') as file:
        data = json.load(file)
        for file in data:
            id = file.split('/')[-1].replace('.txt','')
            test_ids.append(id)
    
    return test_ids, train_val_ids


test_ids, train_val_ids = get_train_test_ids('/home/student/move/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/')

train, val, test = get_interestPoints_from_parts_indices(visit_files, test_ids + train_val_ids)
print(len(train))

print('########################################')
print(len(test))

print('########################################')
print(len(val))
#new_test_ids = match_dataset_to_visit_id(test_ids, train_val_ids, ann_id_to_visit_id)
#print(new_test_ids)

pathh = '/home/student/move/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

with open(pathh+'shuffled_train_file_list.json', 'w') as file:
    json.dump(train, file, indent=4)
    
with open(pathh+'shuffled_val_file_list.json', 'w') as file:
    json.dump(test, file, indent=4)

with open(pathh+'shuffled_test_file_list.json','w') as file:
    json.dump(val, file, indent=4)
