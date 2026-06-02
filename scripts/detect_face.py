import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

INPUT_DIR = 'prepared_dataset'
OUTPUT_DIR = 'detect_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = '../models/blaze_face_short_range.tflite'

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
)

detector = vision.FaceDetector.create_from_options(options)


for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(INPUT_DIR, filename)

    print(f'Processing: {filename}')

    image = cv2.imread(image_path)

    if image is None:
        print('Failed to load image')
        continue

    h, w, _ = image.shape

    # BGR -> RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # face detection
    result = detector.detect(mp_image)

    if not result.detections:
        print('No face detected')
        continue

    # use first detected face
    detection = result.detections[0]

    bbox = detection.bounding_box

    x = int(bbox.origin_x)
    y = int(bbox.origin_y)
    width = int(bbox.width)
    height = int(bbox.height)

    # safety bounds
    x = max(0, x)
    y = max(0, y)

    width = min(width, w - x)
    height = min(height, h - y)

    # crop face
    face_crop = image[y:y + height, x:x + width]

    # resize for ResNet
    face_crop = cv2.resize(face_crop, (224, 224))

    # save face
    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(output_path, face_crop)

    print(f'Saved: {output_path}')

print('\nFace detection completed.')