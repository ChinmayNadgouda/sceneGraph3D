import json
import numpy as np

def matrix_to_string(matrix):
     """Convert a 4x4 matrix to a whitespace-separated string of 16 elements."""
     flat_matrix = matrix.flatten()  # Flatten the matrix into a 1D array
     flat_list = flat_matrix.tolist()  # Convert to list
     string_representation = ' '.join(map(str, flat_list))  # Convert list to a string

     return string_representation

with open('/home/gokul/Downloads/transforms.json') as file:
    json_data = json.load(file)

new_dict = {}
for frame in json_data['frames']:
    id = frame['file_path'].split('/')[-1].split('.')[0]
    new_dict[id] = frame['transform_matrix']
    
sorted_dict = { key: new_dict[key] for key in sorted(new_dict.keys())}
for key, pose in sorted_dict.items():
    string = matrix_to_string(np.array(pose))
    with open('/home/gokul/Downloads/nerf.txt', 'a') as file:
        file.write(string + '\n')