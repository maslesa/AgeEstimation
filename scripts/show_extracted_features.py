import os
import numpy as np
import pandas as pd

FEATURES_DIR = 'extracted_features'
data = []

npy_files = [
    file for file in os.listdir(FEATURES_DIR)
    if file.endswith('.npy')
][:20]

for filename in npy_files:

    file_path = os.path.join(
        FEATURES_DIR,
        filename
    )

    features = np.load(file_path)

    features = features.flatten()

    data.append({
        'image_npy_array': filename,
        'extracted_features': features.tolist()
    })

df = pd.DataFrame(data)
print(df.shape)
print(df)