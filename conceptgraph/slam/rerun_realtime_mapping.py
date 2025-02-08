'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import uuid
from pathlib import Path
import pickle
import gzip
from transformers import AutoModelForCausalLM, AutoTokenizer  
# Third-party imports
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import open3d as o3d
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter
from conceptgraph.Mask3D.eval import main as mask3d_main
from hydra.core.global_hydra import GlobalHydra
import time
from hydra.experimental import initialize, compose
initialize(config_path="../Mask3D/conf", job_name="test_app")  # Initialize Hydra
cfg_mask3d = compose(config_name="config_base_instance_segmentation.yaml")  # Load the Hydra configuration
# Clear the existing instance
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client, \
    consolidate_captions_llava, consolidate_captions_llava_1_6_mistral, get_affordance_label_from_list_for_given_obj
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges_and_captions, 
    measure_time, 
    save_detection_results,
    save_edge_json, 
    save_hydra_config,
    save_obj_json, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
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
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections


import os
from mistralai import Mistral
# Disable torch gradient computation
torch.set_grad_enabled(False)
def get_part_name(object_class:str, affordance_list:str):
    model = "mistral-large-latest"

    client = Mistral(api_key='4hhDbtrs8XGj6XvsYP4YAm3qSl0zH2CO')
    prompt = """You are an agent specialized in identifying the correct affordance label of objects based on a list of affordance labels provided.
        Here is an example input, cabinet, ['pinch_pull','rotate','hook_turn','key_press','foot_push']
        You will be provided with an object class and a list of affordance labels for the object. Your task is to determine the most appropriate label. 
        Your response should be a only the affordance label in the following format ['pinch_pull']

    """
    user_prompt = 'Here is the object class,' + object_class + 'The affordance list are' +  str(affordance_list) + 'Give me the most probable affordance label in rquested format.'

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt + user_prompt,
            },
        ]
    )
    time.sleep(2)
    prompt = """You are an agent specialized in identifying the correct name of an object part based on  the affordance label of the part and object class provided.
    Here is an example input, hook_turn and  door
    You will be provided with an object class and a affordance label for the part. Your task is to determine the most appropriate part name. 
    Your response should be a only the part name in the following format ['handle']

    """
    user_prompt = 'Here is the object class,' + object_class + 'The affordance label is' + str(chat_response.choices[0].message.content) + 'Give me the most probable part name in rquested format.'

    chat_response2 = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt + user_prompt,
            },
        ]
    )
    time.sleep(2)

    return eval(chat_response2.choices[0].message.content)[0], eval(chat_response.choices[0].message.content)[0]

def get_part_tasks(object_class, part_name, clip_model, clip_tokenizer):
    model = "mistral-large-latest"

    client = Mistral(api_key='4hhDbtrs8XGj6XvsYP4YAm3qSl0zH2CO')
    prompt = """You are an agent specialized in decribing at least five uses for a given part name and its object.
    Here is an example input, handle and  cabinet
    You will be provided with an object class and a name for the part. Your task is to describe the five most appropriate tasks that can be done using the part. 
    Your response should be a only the tasks in the following format ['task1', 'task2', .... ]
    """
    user_prompt = 'Here is the object class,' + object_class + 'The name for the part is ' +  part_name + 'Give me the tasks in requested format.'

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt + user_prompt,
            },
        ]
    )
    time.sleep(2)

    tasks = eval(chat_response.choices[0].message.content)
    tokens = clip_tokenizer(tasks)
    tokens = tokens.to('cuda:0')

    # Get the text embeddings from CLIP model
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)

    # Now, `text_features` contains the embeddings for each task.
    # Combine them into a single embedding by averaging the embeddings.
    combined_embedding = text_features.mean(dim=0)

    # Print the final combined embedding
    print(combined_embedding.shape)  # Should be a vector of shape (512,)
    print(combined_embedding) 
    return combined_embedding

    
# A logger for this file
@hydra.main(config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):
    tracker = MappingTracker()
    
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    #orr.connect('141.58.226.203:9876')

    orr.spawn()


    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", 
            #    entity="concept-graphs",
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)

    # Initialize the dataset
    scene_iddd = str(cfg.scene_id)
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=scene_iddd,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()
    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # For visualization
    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, scene_iddd, cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, scene_iddd, cfg.detections_exp_suffix, make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        #sam_predictor = SAM('sam_l.pt') 
        sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", "laion400m_e31"
        )
        # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        #     "ViT-H-14", "laion2b_s32b_b79k"
        # )
        #clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())

        # openai_client = get_openai_client()
        # model_id = "/home/gokul/ConceptGraphs/llava-v1.5-7b/models--llava-hf--llava-1.5-7b-hf/snapshots/fa3dd2809b8de6327002947c3382260de45015d4"
        # model_id = "llava-hf/llava-1.5-7b-hf"
        #model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        #model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
        # model_id = "/media/gokul/Elements1/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
        # cache_dir = "/media/gokul/Elements1/huggingface_cache" 
        # processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        # bnb_config = BitsAndBytesConfig(
        #             load_in_4bit=True,
        #             bnb_4bit_use_double_quant=True,
        #             bnb_4bit_quant_type="nf4",
        #             bnb_4bit_compute_dtype=torch.bfloat16
        #             )
        # model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, quantization_config=bnb_config,
        #             use_flash_attention_2=True, cache_dir=cache_dir ) 
        processor = None
        model = None
        
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time_sequence("frame", frame_idx)

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        depth_path = Path(dataset.depth_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Do initial object detection
            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            
            # Make the edges
            labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_edges, (model,processor), depth_path)

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
                "captions": captions,
            }

            raw_gobs = results

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                # if no detections, throw an error
                raise FileNotFoundError(f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / color_path.stem} or \n{det_exp_pkl_path / f'{int(color_path.stem):06}'}.")

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
        
        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
        orr_log_depth_image(depth_tensor)
        orr_log_vlm_image(vis_save_path_for_vlm)
        orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # this helps make sure things like pillows on couches are separate objects
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0: # no detections, skip
            continue

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                })
            continue 

        ### compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, 
            detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
        )

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )
        # fix the class names for objects
        # they should be the most popular name, not the first name
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Clean up outlier edges
        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            obj1_class_name = objects[curr_obj1_idx]['class_name'] 
            obj2_class_name = objects[curr_obj2_idx]['class_name']
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )

        # Merging
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=cfg["make_edges"],
                map_edges=map_edges
            )
        




        #Integrating Mask3D
        if is_final_frame:
            part_list = {
                0:'exclude2',
                1:'hook_turn',
                2:'exclude',
                3:'hook_pull',
                4:'key_press',
                5:'rotate',
                6:'foot_push',
                7:'unplug',
                8:'plug_in',
                9:'pinch_pull',
                10:'tip_push'
            }
            part_obj_classes = ObjectClasses(
                classes_file_path='/home/gokul/ConceptGraphs/concept-graphs/conceptgraph/part-object-classes.txt', 
                bg_classes=['wall','floor','ceiling'], 
                skip_bg=False
            )
            valid_objects = part_obj_classes.get_classes_arr()
            count = 0
            max_obj_count = 0
            for obj in objects:
                if max_obj_count > obj['new_counter']:
                    pass
                else:
                    max_obj_count = obj['new_counter']
            
            new_obj_count = 1
            for obj in objects:
                if obj and len(obj['pcd'].points) > 500 and obj['class_name'] in valid_objects:
                  count += 1
                  temp_obj = o3d.geometry.PointCloud()
                  temp_obj.points = obj['pcd'].points
                  temp_obj.colors = obj['pcd'].colors
                  temp_obj.estimate_normals(
                     search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)  # KNN search
                  )
                  masks_scores = mask3d_main(cfg_mask3d, np.asarray(temp_obj.points), np.asarray(temp_obj.colors),  np.asarray(temp_obj.normals), obj['class_name'])
                  obj_points = np.asarray(obj['pcd'].points)
                  obj_colors = np.asarray(obj['pcd'].colors)
                  affordance_labels = []
                  for key in masks_scores.keys():
                      affordance_labels.append(part_list[key])
                  part_name, part_label = get_part_name(obj['class_name'], affordance_labels)
                  for key, value in masks_scores.items():
                        if part_list[key] == part_label:
                            masks = np.array(value['masks'])

                            # Aggregation using logical OR (binary mask)
                            mask = np.any(masks, axis=0).astype(bool)
                            #mask = value['masks'][0].astype(bool)
                            inverse_mask = ~ value['masks'][0].astype(bool)
                            new_points = obj_points[mask]
                            if not len(new_points) > 4:
                                continue
                            new_colors = obj_colors[mask]
                            obj['pcd'].points = o3d.utility.Vector3dVector(obj_points[inverse_mask])
                            obj['pcd'].colors = o3d.utility.Vector3dVector(obj_colors[inverse_mask]) 
                            new_obj = copy.deepcopy(obj)
                            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_points)
                            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_colors)
                            new_obj['new_counter'] = max_obj_count + new_obj_count
                            new_obj['class_name'] = part_name
                            new_obj['class_id'] = 200 + int(key)
                            new_obj['num_detections'] = 1
                            new_obj['n_points'] = len(new_points)
                            new_obj['num_obj_in_class'] = 1
                            new_obj['curr_obj_num'] = max_obj_count + new_obj_count
                            new_obj['bbox'] = new_obj['pcd'].get_oriented_bounding_box(robust=True)
                            print(new_obj)
                            print(get_part_tasks(obj['class_name'], part_name, clip_model, clip_tokenizer))
                            new_obj_count += 1
                            objects.append(new_obj)
                            # model_name_llm = "mistralai/Mistral-7B-v0.1"  # Ensure you have enough VRAM  
                            # tokenizer = AutoTokenizer.from_pretrained(model_name_llm)  
                            # model = AutoModelForCausalLM.from_pretrained(model_name_llm, torch_dtype=torch.float16, device_map="auto")  
                            # question =  'For the object' + obj['class_name'] + ', what would be the name of its part/ functionally interactive element if its label is '+new_obj['class_name']
                            
                            # inputs = tokenizer(question, return_tensors="pt").to("cuda")  
                            # output = model.generate(**inputs, max_new_tokens=100)  
                            # print('LLM answer:',tokenizer.decode(output[0], skip_special_tokens=True))
                        

        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path
            )
        
        if cfg.vis_render:
            # render a frame, if needed (not really used anymore since rerun)
            vis_render_image(
                objects,
                obj_classes,
                obj_renderer,
                image_original_pil,
                adjusted_pose,
                frames,
                frame_idx,
                color_path,
                cfg.obj_min_detections,
                cfg.class_agnostic,
                cfg.debug_render,
                is_final_frame,
                cfg.exp_out_path,
                cfg.exp_suffix,
            )

        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            # save the pointcloud
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True
            )

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                })
    # LOOP OVER -----------------------------------------------------
    
    #Consolidate captions
    # for object in objects:
    #     obj_captions = object['captions'][:20]
    #     consolidated_caption = consolidate_captions_llava_1_6_mistral((model,processor), obj_captions)
    #     object['consolidated_caption'] = consolidated_caption

    handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

    # Save the pointcloud
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges
        )

    if cfg.save_json:
        save_obj_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects
        )
        
        save_edge_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects,
            edges=map_edges
        )

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()

if __name__ == "__main__":
    main()
