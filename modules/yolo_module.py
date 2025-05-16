from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path="yolov8l.pt"):
        self.model = YOLO(model_path)

    # Modify detect() in yolo_module.py
    def detect(self, image, conf_threshold=0.4):
        results = self.model(image)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] < conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                label = self.model.names[cls]
                detections.append({
                    "label": label,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0])
                })
        return detections

