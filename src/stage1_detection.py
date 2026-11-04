from ultralytics import YOLO
import cv2


class WasteDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not found")

        results = self.model(image, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = self.model.names[class_id]

                detections.append({
                    "material": label,
                    "confidence": confidence
                })

        return detections