import os
from datetime import datetime
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

OUTPUT_DIR = 'detect_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = '../models/blaze_face_short_range.tflite'

def detect_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print('Failed to load image')
        return

    h, w, _ = image.shape

    # convert to RGB image, because mediapipe uses RGB, but OpenCV uses BGR
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # converting to MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # model configuration
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )

    # creating detector
    detector = vision.FaceDetector.create_from_options(options)

    result = detector.detect(mp_image)

    if not result.detections:
        print("No faces detected")
        return

    for i, detection in enumerate(result.detections):
        bbox = detection.bounding_box

        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        width = int(bbox.width)
        height = int(bbox.height)

        # make sure the face stays in the rectangle
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)

        # creating rectangle
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # crop the face
        face_crop = image[y:y + height, x:x + width]

        # save cropped face
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        face_path = os.path.join(OUTPUT_DIR, f"face_{i}_{timestamp}.jpg")
        cv2.imwrite(face_path, face_crop)

    print("Detection done.")
    print(f"Saved in: {OUTPUT_DIR}")

detect_face('./detect_test/2.jpg')
