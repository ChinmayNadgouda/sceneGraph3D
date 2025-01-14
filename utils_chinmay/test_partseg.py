"""
Author: Benny
Date: Nov 2019
"""
#import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import argparse
import os
from data_utils.SceneFun3dDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# seg_classes = {
#  'tip_push':[0,10],
#     'hook_turn':[1,11],
#     'hook_pull':[3,13],
#     'key_press':[4,14],
#     'rotate':[5,15],
#     'foot_push':[6,16],
#     'unplug':[7,17],
#     'plug_in':[8,18],
#     'pinch_pull':[9,19]
# }
seg_classes = {
        'tip_push':[0,1],
            'hook_turn':[2,3],
            'hook_pull':[4,5],
            'key_press':[6,7],
            'rotate':[8,9],
            'foot_push':[10,11],
            'unplug':[12,13],
            'plug_in':[14,15],
            'pinch_pull':[16,17]
        }
# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()
def generate_color_map(num_colors):
    # Generate a color map with evenly spaced colors in RGB space
    colors = {}
    for i in range(num_colors):
        # Use a circular color wheel or a gradient approach for distinct colors
        r = int(255 * (np.sin(2 * np.pi * i / num_colors) + 1) / 2)
        g = int(255 * (np.sin(2 * np.pi * (i + 1) / num_colors) + 1) / 2)
        b = int(255 * (np.sin(2 * np.pi * (i + 2) / num_colors) + 1) / 2)
        colors[i] = [r, g, b]
    return colors

# Generate the color map for 20 colors


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg_firstsSuccessfulRun/' + args.log_dir
    iii=0
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='val', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 9
    num_part = 18

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()   
            print(type(points), type(label), type(target))  
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            # pcd2 = o3d.geometry.PointCloud()

            # # Set the points (x, y, z)
            # pcd2.points = o3d.utility.Vector3dVector(points[0, :, :3].detach().cpu().numpy() )  # Only use the first 3 columns (x, y, z)

            # # Set the colors for each point (r, g, b)

            # # 6. Visualize the point cloud
            # vis2 = o3d.visualization.VisualizerWithEditing()
            # vis2.create_window()
            # vis2.add_geometry(pcd2)
            # vis2.run()
            # #o3d.visualization.draw_geometries([pcd2])
            points = points.transpose(2, 1)
            # print(122,points.shape)
            # # part1 = points[:,:,:128]
            # # points2 = torch.concat((part1, part1, part1, part1, part1, part1, part1, part1, part1, part1, part1, part1, part1, part1, part1, part1), dim=2)
            # # points = 
            # z_values = points[:, 0, :]  # Shape: (24, 2048)

            # # Get the indices that would sort the z-values for each batch
            # sorted_indices = torch.argsort(z_values, dim=1, descending=True)  # Sort in descending order based on z-values

            # # Use these indices to reorder the original point cloud along the last dimension
            # sorted_point_cloud = torch.gather(points, dim=2, index=sorted_indices.unsqueeze(1).expand(-1, 6, -1))
            # part1 = sorted_point_cloud[:,:,:1024]
            # points = torch.concat((part1,part1), dim=2)
            # print(135,points.shape)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                if iii < 100:
                    #print(seg_pred)
                    pointsss = points    
                    points23 = pointsss.transpose(2,1)
                    seg_probabilities = F.softmax(seg_pred[0], dim=1)  # Shape [2048, 50]
                    
                    # 2. Get the predicted class for each point (index of the maximum probability)
                    predicted_classes = torch.argmax(seg_probabilities, dim=1)  # Shape [2048]
                    predicted_classes = predicted_classes.cpu()
                    print(predicted_classes)
                    
                    print('uni',torch.unique(predicted_classes))
                    if (torch.equal(torch.tensor([2, 3]), torch.unique(predicted_classes))) or (torch.equal(torch.tensor([4, 5]), torch.unique(predicted_classes))) or  (torch.equal(torch.tensor([8, 9]), torch.unique(predicted_classes))) or  (torch.equal(torch.tensor([10, 11]), torch.unique(predicted_classes)))  or  (torch.equal(torch.tensor([16, 17]), torch.unique(predicted_classes))):
                        

                        # 3. Convert points to numpy array (you can access .numpy() since it's on CPU)
                        points_np = points23[0].detach().cpu().numpy()  # Shape [2048, 6]
                        #print(points_np)

                        # 4. Map each predicted class to a color (let's use a simple color map for now)
                        # For simplicity, let's map the class index to a color. Here we use a basic color map.
                        # You can modify this as per your requirements (e.g., using a predefined colormap)
                        
                        colors = {
                            0: [0, 0, 0],
                            1: [0.8, 0.8, 0.8],
                            2: [0, 0, 0],
                            3: [0.8, 0.8, 0.8],
                            4: [0, 0, 0],
                            5: [0.8, 0.8, 0.8],
                            6: [0, 0, 0],
                            7: [0.8, 0.8, 0.8],
                            8: [0, 0, 0],
                            9: [0.8, 0.8, 0.8],
                            10: [0, 0, 0],
                            11: [0.8, 0.8, 0.8],
                            12: [0, 0, 0],
                            13: [0.8, 0.8, 0.8],
                            14: [0, 0, 0],
                            15: [0.8, 0.8, 0.8],
                            16: [0, 0, 0],
                            17: [0.8, 0.8, 0.8]
                        }

                        # # Create a list of colors for each point based on its predicted class
                        point_colors = np.array([colors[c.int().tolist()] for c in predicted_classes])  # RGB colors (0-1 range)
                        # Create a 3D plot
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')

                        # Scatter plot: points, c=colors (for RGB color map)
                        scatter = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=point_colors)

                        # Add labels (optional)
                        ax.set_xlabel('X Label')
                        ax.set_ylabel('Y Label')
                        ax.set_zlabel('Z Label')

                        # Show the plot
                        
                        output_file = '/home/chinmayn/points_and_colors.txt'

                        # Open the file in write mode
                        with open(output_file, 'w') as file:
                            for i in range(len(points_np)):
                                # Write each point's coordinates and color in the format: x y z r g b
                                file.write(f"{points_np[i, 0]} {points_np[i, 1]} {points_np[i, 2]} {point_colors[i, 0]} {point_colors[i, 1]} {point_colors[i, 2]} {predicted_classes[i]}\n")
                        
                        plt.show()
                        # # 5. Create an Open3D PointCloud object
                        # pcd = o3d.geometry.PointCloud()

                        # # Set the points (x, y, z)
                        # pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])  # Only use the first 3 columns (x, y, z)

                        # Set the colors for each point (r, g, b)
                        #pcd.colors = o3d.utility.Vector3dVector(point_colors)

                        #6. Visualize the point cloud
                        # vis = o3d.visualization.VisualizerWithEditing()
                        # vis.create_window()
                        # vis.add_geometry(pcd)
                        # vis.run()
                        # o3d.visualization.draw_geometries([pcd])
                        iii += 1
                    
                    #exit()
                   
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
