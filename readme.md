# Face detection system

### Requirements
1. Virtual environment activation:
```
# Windows
.venv/Scripts/activate

# Linux
source .venv/Scripts/activate
```

2. Download MediaPipe model:
[Blaze face short_range](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite).


3. Create *models* directory and put model there.
```
models/blaze_face_short_range.tflite
```

4. Run the script (image examples are given in detect_test directory)
```
cd scripts
python detect_face.py
```