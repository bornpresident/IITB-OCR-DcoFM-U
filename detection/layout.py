from doclayout_yolo import YOLOv10
import torchvision
import cv2
import pathlib

current_dir = pathlib.Path(__file__).parent.absolute()
model_path = str(current_dir) + "/model/doclayout_yolo_dsb_1024.pt"
print(model_path)
model = YOLOv10(model_path)
class_names = {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}

def get_page_layout(image_path, layout_annotated_image_path, device = 'cpu'):
    det_res = model.predict(
    image_path,         # Image to predict
    imgsz = 1024,       # Prediction image size
    conf = 0.1,         # Confidence threshold
    iou = 0.0001,       # NMS Threshold ??
    device = device,    # Device to use (e.g., 'cuda:0' or 'cpu')
    save = False,       # No need to save annotated image
    verbose = False)
    dets = []
    for entry in det_res:
        bboxes = entry.boxes.xyxy
        classes = entry.boxes.cls
        conf = entry.boxes.conf
        keep = torchvision.ops.nms(bboxes, conf, iou_threshold = 0.1)
        bboxes = bboxes[keep].cpu().numpy()
        classes = classes[keep].cpu().numpy()
        conf = conf[keep].cpu().numpy()
        for i in range(len(bboxes)):
            box = bboxes[i]
            dets.append([classes[i], [int(box[0]), int(box[1]), int(box[2]), int(box[3])]])
        draw_bboxes(image_path, dets, layout_annotated_image_path)
    return dets

def draw_bboxes(image_path, dets, layout_annotated_image_path):
    image = cv2.imread(image_path)
    for det in dets:
        cls = det[0]
        if cls == 5:
            # class_name = 'table'
            color = (128, 127, 0)
        elif cls == 8:
            # class_name = 'equation'
            color = (0, 0, 255)
        elif cls == 3:
            # class_name = 'figure'
            color = (255, 0, 0)
        else:
            # class_name = 'text'
            color = (0, 255, 0)
        bbox = det[1]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.imwrite(layout_annotated_image_path, image)