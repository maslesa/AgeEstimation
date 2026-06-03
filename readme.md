# Age estimation on Adience dataset using XGBoost

### Requirements
1. **Create and activate virtual environment:**
```
# Create venv
python -m venv .venv

# Windows activation
.venv/Scripts/activate

# Linux activation
source .venv/Scripts/activate
```

2. **Install dependencies:**
```
pip install -r requirements.txt
```

3. **Download MediaPipe model:**
[Blaze face short_range](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite).


4. **Create *models* directory and put model there.**
```
models/blaze_face_short_range.tflite
```

5. **Run the scripts**
```
cd scripts
python 01_prepare_dataset.py
python 02_detect_face.py
python 03_extract_features.py
python 04_create_final_dataset.py
```

6. **Train and test model (open 05_age_estimation_xgboost.ipynb and run all cells).**