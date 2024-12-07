from ultralytics import YOLO, SAM

from conceptgraph.utils.general_utils import (
    ObjectClasses )

obj_classes = ObjectClasses(
        classes_file_path='/home/student/ConceptGraph/sceneGraph3D/conceptgraph/scannet200_classes.txt', 
        bg_classes=['wall','floor','ceiling'], 
        skip_bg=False
    )

detection_model = YOLO("yolov8l-world.pt")
#sam_predictor = SAM('sam_l.pt') 
sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM

detection_model.set_classes(obj_classes.get_classes_arr())

color_path = '/home/student/Downloads/frame000000.jpg'

results = detection_model(color_path, conf=0.1, verbose=False)

confidences = results[0].boxes.conf.cpu().numpy()
detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
xyxy_tensor = results[0].boxes.xyxy
xyxy_np = xyxy_tensor.cpu().numpy()

print(detection_class_labels)
# if there are detections,
# Get Masks Using SAM or MobileSAM
# UltraLytics SAM
# if xyxy_tensor.numel() != 0:
#     sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
#     masks_tensor = sam_out[0].masks.data

#     masks_np = masks_tensor.cpu().numpy()
# else:
#     masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)