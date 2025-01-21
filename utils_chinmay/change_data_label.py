import numpy as np
import os
sett = set()
count = 0
folder_path = '/media/gokul/Elements/Chinmay_Ubuntu/Mask3D/processed/scannet200/'
for root, dirs, files in os.walk(folder_path):
        if len(dirs) > 2:
            for dir in dirs:
                if dir == 'train' and root == '/media/gokul/Elements/Chinmay_Ubuntu/Mask3D/processed/scannet200/' :
                    for root, dirs, files in os.walk(folder_path+dir):
                        for file in files:
                            count += 1
                            data = np.load(folder_path+dir+'/'+file)
                            second_last_col = data[:, -2]
                            conditions = [
                                second_last_col < 120,   # Condition 1: Values less than 120
                                second_last_col == 120,  # Condition 2: Value exactly 120
                                (second_last_col >= 121) & (second_last_col <= 129)  # Condition 3: Values between 121 and 129
                            ]

                            # Define corresponding values
                            choices = [
                                2,                       # Values less than 120 map to 2
                                10,                      # Value 120 maps to 10
                                second_last_col - 120     # Values 121 to 129 map to 1 to 9
                            ]
                            second_last_col = np.select(conditions, choices, second_last_col)


                            # Assign the modified column back to the original array
                            data[:, -2] = second_last_col
                            for col in second_last_col:
                                sett.add(col)
                            np.save('/home/gokul/Mask3D/data/processed/train/'+file, data)

print(count, sett)