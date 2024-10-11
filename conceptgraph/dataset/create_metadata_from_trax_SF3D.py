import json

file_path = "/home/gokul/ConceptGraphs/datasets/record3d_scans/room1_preprocessed"
f = open(file_path+"/traj.txt", "r")

new_json = {"initPose":[0,0,0,1,0,0,0],"dh":192,"w":1920,"dw":256,"poses":[],"cameraType":1,"h":1440,"fps":30,"K":[1457.38037109375,0,0,0,1457.38037109375,0,721.4085693359375,929.7322998046875,1]}

for line in f.readlines():
    line_array = line.split()
    new_json['poses'].append(line_array)

print(new_json)
f.close()

save_file = open(file_path+"/metadata", "w")
json.dump(new_json, save_file, indent = 6)
save_file.close()