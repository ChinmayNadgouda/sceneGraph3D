import cv2
import os
# import PyQt5

# # Set the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
# pyqt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugin_path

import copy
import json
import os
import pickle
import gzip
import argparse

import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip
import hydra
from conceptgraph.dataset.datasets_common import get_dataset
from pathlib import Path
from PIL import Image
from conceptgraph.Mask3D.eval import main as mask3d_main
import distinctipy

from hydra.experimental import initialize, compose
initialize(config_path="../Mask3D/conf", job_name="test_app")  # Initialize Hydra
cfg = compose(config_name="config_base_instance_segmentation.yaml")  # Load the Hydra configuration
from omegaconf import DictConfig
from ultralytics import YOLO, SAM
import supervision as sv
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    make_vlm_edges_and_captions, 
)
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.utils.ious import mask_subtract_contained

# from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.utils.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects

@hydra.main(config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
def get_config(cfg: DictConfig):
    return cfg
def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    
    parser.add_argument("--no_clip", action="store_true", 
                        help="If set, the CLIP model will not init for fast debugging.")
    
    # To inspect the results of merge_overlap_objects
    # This is mainly to quickly try out different thresholds
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    
    return parser

def load_result(result_path):
    # check if theres a potential symlink for result_path and resolve it
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary! other types are not supported!")
    
    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    bg_objects = MapObjectList()
    bg_objects.extend(obj for obj in objects if obj['is_background'])
    if len(bg_objects) == 0:
        bg_objects = None
    class_colors = results['class_colors']
        
    
        
    return objects, bg_objects, class_colors

def main(args):
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path
    
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)
        
        if result_path is None:
            print("Only visualizing the pointcloud...")
            o3d.visualization.draw_geometries([global_pcd])
            exit()
        
    objects, bg_objects, class_colors = load_result(result_path)
    
    if args.edge_file is not None:
        # Load edge files and create meshes for the scene graph
        scene_graph_geometries = []
        with open(args.edge_file, "r") as f:
            edges = json.load(f)
        
        classes = objects.get_most_common_class()
        colors = [class_colors[str(c)] for c in classes]
        obj_centers = []
        for obj, c in zip(objects, colors):
            pcd = obj['pcd']
            bbox = obj['bbox']
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            extent = bbox.get_max_bound()
            extent = np.linalg.norm(extent)
            # radius = extent ** 0.5 / 25
            radius = 0.10
            obj_centers.append(center)

            # remove the nodes on the ceiling, for better visualization
            ball = create_ball_mesh(center, radius, c)
            scene_graph_geometries.append(ball)
            
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']

            line_mesh = LineMesh(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [1, 0, 0],
                radius=0.02
            )

            scene_graph_geometries.extend(line_mesh.cylinder_segments)
    
    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", "laion400m_e31")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    if bg_objects is not None:
        indices_bg = []
        for obj_idx, obj in enumerate(objects):
            if obj['is_background']:
                indices_bg.append(obj_idx)
        # indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        # objects.extend(bg_objects)
        
    # Sub-sample the point cloud for better interactive experience
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        # pcd = pcd.voxel_down_sample(0.05)
        objects[i]['pcd'] = pcd
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # Get the color for each object when colored by their class
    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        # Get the most common class for this object as the class
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
    
    # Set the title of the window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    if result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)

    # Add geometry to the scene
    for geometry in pcds + bboxes:
        vis.add_geometry(geometry)
        
    main.show_bg_pcd = True
    def toggle_bg_pcd(vis):
        if bg_objects is None:
            print("No background objects found.")
            return
        
        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)
        
        main.show_bg_pcd = not main.show_bg_pcd
        
    main.show_global_pcd = False
    def toggle_global_pcd(vis):
        if args.rgb_pcd_path is None:
            print("No RGB pcd path provided.")
            return
        
        if main.show_global_pcd:
            vis.remove_geometry(global_pcd, reset_bounding_box=False)
        else:
            vis.add_geometry(global_pcd, reset_bounding_box=False)
        
        main.show_global_pcd = not main.show_global_pcd
        
    main.show_scene_graph = False
    def toggle_scene_graph(vis):
        if args.edge_file is None:
            print("No edge file provided.")
            return
        
        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        
        main.show_scene_graph = not main.show_scene_graph
        
    def color_by_class(vis):
        for i in range(len(objects)):
            pcd = pcds[i]
            obj_class = object_classes[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    class_colors[str(obj_class)],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_rgb(vis):
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]['pcd'].colors
        
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_instance(vis):
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    instance_colors[i, :3],
                    (len(pcd.points), 1)
                )
            )
            
        for pcd in pcds:
            vis.update_geometry(pcd)
        
    def color_by_clip_sim(vis):
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        text_query = input("Enter your query: ")
        text_queries = [text_query]
        
        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

        max_prob_object = objects[max_prob_idx]
        
        # Estimate normals
        # Sending most probable object to mask3d
        pcd_to_eval = max_prob_object['pcd']
        pcd_to_eval.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1,  # Search radius
                max_nn=30    # Maximum nearest neighbors
            )
        )
        mask3d_main(cfg, np.asarray(pcd_to_eval.points), np.asarray(pcd_to_eval.colors),  np.asarray(pcd_to_eval.normals))
        # count_img = 0
        # dataset = get_dataset(
        #         dataconfig='/home/gokul/ConceptGraphs/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml',
        #         start=0,
        #         end=-1,
        #         stride=10,
        #         basedir='/home/gokul/ConceptGraphs/datasets/Replica',
        #         sequence='room0',
        #         desired_height=680,
        #         desired_width=1200,
        #         device="cpu",
        #         dtype=torch.float,
        #     )
        # for img in max_prob_object['image_idx']:
        #     frame_idx = img
        #     scene_iddd = 'room0'
            
        #     color_path = Path(dataset.color_paths[frame_idx])
        #     depth_path = Path(dataset.depth_paths[frame_idx])
        #     image_original_pil = Image.open(color_path)
        #     # color and depth tensors, and camera instrinsics matrix
        #     color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        #     # Covert to numpy and do some sanity checks
        #     depth_tensor = depth_tensor[..., 0]
        #     depth_array = depth_tensor.cpu().numpy()
        #     color_np = color_tensor.cpu().numpy() # (H, W, 3)
        #     image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        #     assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        #     color_path = str(max_prob_object['color_path'][count_img])
        #     count_img += 1
        #     color_path = color_path.replace('student','gokul')
        #     color_path = color_path.replace('ConceptGraph','ConceptGraphs')
        #     #print(max_prob_object)
        #     #cfg = get_config()
        #     #cfg = process_cfg(cfg)
        #     obj_classes = ObjectClasses(
        #     classes_file_path='classes.txt', 
        #     bg_classes=["wall", "floor", "ceiling"], 
        #     skip_bg=True
        #     )
        #     detection_model = YOLO('yolov8l-world.pt')
        #     sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM
        #     detection_model.set_classes(obj_classes.get_classes_arr())
        #     results = detection_model.predict(color_path, conf=0.1, verbose=False)
        #     print(results[0].boxes.cls)
        #     confidences = results[0].boxes.conf.cpu().numpy()
        #     detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        #     detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
        #     xyxy_tensor = results[0].boxes.xyxy
        #     xyxy_np = xyxy_tensor.cpu().numpy()
        #     j=0
        #     print(detection_class_ids)
        #     for part in detection_class_labels:
        #         print(part)
        #         x,y,x1,y1 = map(float,xyxy_np[j])
        #         print(x, y, x1, y1)
        #         print(xyxy_np[j])
        #         # cropped_img2 = pil_image.crop((x, y, x1, y1))
        #         # cropped_image2 = cv2.cvtColor(np.array(cropped_img2), cv2.COLOR_RGB2BGR)
        #         # cv2.imshow("cropped", cropped_image2)
        #         # cv2.imwrite("/home/gokul/imgess/tryCI_pilchin2"+str(i)+str(j)+".jpg", cropped_image2)
        #         # print("/home/gokul/CI"+str(i)+".jpg")
        #         j+=1
        #     if xyxy_tensor.numel() != 0:
        #             sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
        #             masks_tensor = sam_out[0].masks.data

        #             masks_np = masks_tensor.cpu().numpy()
        #     else:
        #         masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

        #     curr_det = sv.Detections(
        #             xyxy=xyxy_np,
        #             confidence=confidences,
        #             class_id=detection_class_ids,
        #             mask=masks_np,
        #         )
        #     print(curr_det)
        # # Make the edges
        # labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, False, (None,None), depth_path)

        # image_crops, image_feats, text_feats = compute_clip_features_batched(
        #     image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), 'cuda')

        # # increment total object detections

        # # Save results
        # # Convert the detections to a dict. The elements are in np.array
        # results = {
        #     # add new uuid for each detection 
        #     "xyxy": curr_det.xyxy,
        #     "confidence": curr_det.confidence,
        #     "class_id": curr_det.class_id,
        #     "mask": curr_det.mask,
        #     "classes": obj_classes.get_classes_arr(),
        #     "image_crops": image_crops,
        #     "image_feats": image_feats,
        #     "text_feats": text_feats,
        #     "detection_class_labels": detection_class_labels,
        #     "labels": labels,
        #     "edges": edges,
        #     "captions": captions,
        # }
        # raw_gobs = results
        # # get pose, this is the untrasformed pose.
        # unt_pose = dataset.poses[frame_idx]
        # unt_pose = unt_pose.cpu().numpy()

        # # Don't apply any transformation otherwise
        # adjusted_pose = unt_pose

        # # resize the observation if needed
        # resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # # filter the observations
        # filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
        #     skip_bg=True,
        #     BG_CLASSES=['wall'],
        #     mask_area_threshold=25,
        #     max_bbox_area_ratio=0.5,
        #     mask_conf_threshold=0.25,
        # )

        # gobs = filtered_gobs

        # if len(gobs['mask']) == 0: # no detections in this frame
        #     print('No detectionsss')

        # # this helps make sure things like pillows on couches are separate objects
        # gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        # obj_pcds_and_bboxes = detections_to_obj_pcd_and_bbox(
        #     depth_array=depth_array,
        #     masks=gobs['mask'],
        #     cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
        #     image_rgb=image_rgb,
        #     trans_pose=adjusted_pose,
        #     min_points_threshold=cfg.min_points_threshold,
        #     spatial_sim_type=cfg.spatial_sim_type,
        #     obj_pcd_max_points=cfg.obj_pcd_max_points,
        #     device='cuda',
        # )

        # for obj in obj_pcds_and_bboxes:
        #     if obj:
        #         obj["pcd"] = init_process_pcd(
        #             pcd=obj["pcd"],
        #             downsample_voxel_size=cfg["downsample_voxel_size"],
        #             dbscan_remove_noise=cfg["dbscan_remove_noise"],
        #             dbscan_eps=cfg["dbscan_eps"],
        #             dbscan_min_points=cfg["dbscan_min_points"],
        #         )
        #         obj["bbox"] = get_bounding_box(
        #             spatial_sim_type=cfg['spatial_sim_type'], 
        #             pcd=obj["pcd"],
        #         )
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"location xyz: {max_prob_object['bbox'].center}")
        
        for i in range(len(objects)):
            pcd = pcds[i]
            map_colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[i, 0].item(),
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)
        
    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("S"), toggle_global_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("G"), toggle_scene_graph)
    
    # Render the scene
    vis.run()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

'''

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_ram_class_ram_stride50_no_bg__ram_class_ram_stride50_no_bg_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_ram__yolo_class_ram_stride50_no_bg4__ram_yolo_class_ram_stride50_no_bg_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz


python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_ram_class_ram_stride50_no_bg__TEST_ram_class_ram_stride50_no_bg_overlap_maskconf0.25_simsum1.2_dbscan.1.pkl.gz


python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/pcd_saves/full_pcd_scannet200_class_ram_stride50_yes_bg2_mapping_scannet200_class_ram_stride50_yes_bg2.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_38/full_pcd_s_mapping_yes_bg_38.pkl.gz

python scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_39/full_pcd_s_mapping_yes_bg_39_post.pkl.gz


python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_40/full_pcd_s_mapping_yes_bg_40_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_41/full_pcd_s_mapping_yes_bg_41_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_42/full_pcd_s_mapping_yes_bg_42_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/exp_s_mapping_yes_bg_43/full_pcd_s_mapping_yes_bg_43_post.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/office0/exps/s_mapping_yes_bg_multirun_45/full_pcd_s_mapping_yes_bg_multirun_45.pkl.gz


python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/room0/exps/s_mapping_yes_bg_multirun_45/full_pcd_s_mapping_yes_bg_multirun_45.pkl.gz

python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py --result_path /home/kuwajerw/new_local_data/new_replica/Replica/office0/exps/s_mapping_yes_bg_multirun_45/full_pcd_s_mapping_yes_bg_multirun_45.pkl.gz



python concept-graphs/conceptgraph/scripts/streamlined_detections.py

kernprof -l concept-graphs/conceptgraph/slam/streamlined_mapping.py
'''

