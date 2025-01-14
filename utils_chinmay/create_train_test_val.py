import os
import json
import random
from sklearn.model_selection import train_test_split

# Define the root directory where your folders are
root_dir = '/home/student/part_ob'  # Update this with your actual folder path

# Initialize a list to store filenames
file_list = []

# Walk through the root directory and collect filenames
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Skip the "metadata" folder
    if 'meta_data' in dirpath or 'train_test_split' in dirpath:
        continue

    # Collect filenames in the current folder
    for filename in filenames:
        if filename.endswith('.txt'):
           # Assuming you want to include all file types, adjust extension as needed
           file_list.append(os.path.join(dirpath, filename))

# Shuffle the list to randomize the order
random.shuffle(file_list)

# Define the split ratios for train, validation, and test
train_size = 0.65  # 70% for training
val_size = 0.2   # 15% for validation
test_size = 0.15  # 15% for testing

# Split the data into train, test, and validation
train_files, temp_files = train_test_split(file_list, train_size=train_size, random_state=42)
val_files, test_files = train_test_split(temp_files, train_size=val_size / (val_size + test_size), random_state=42)

# Optionally, print the splits
print(f"Total files: {len(file_list)}")
print(f"Train files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files: {len(test_files)}")
with open('/home/chinmayn/old_files/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/shuffled_train_file_list.json', 'w') as train_json:
    json.dump(train_files, train_json, indent=4)

with open('/home/chinmayn/old_files/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/shuffled_val_file_list.json', 'w') as val_json:
    json.dump(val_files, val_json, indent=4)

with open('//home/chinmayn/old_files/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/shuffled_test_file_list.json', 'w') as test_json:
    json.dump(test_files, test_json, indent=4)
# If you need to save the splits, you can write them to text files or handle them as needed
# For example, saving to text files:
#with open('train_files.txt', 'w') as f:
#    for file in train_files:
#        f.write(f"{file}\n")

#with open('val_files.txt', 'w') as f:
#    for file in val_files:
#        f.write(f"{file}\n")

#with open('test_files.txt', 'w') as f:
#    for file in test_files:
#        f.write(f"{file}\n")
