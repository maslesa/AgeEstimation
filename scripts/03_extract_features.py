import os
import pandas as pd
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

INPUT_DIR = 'detect_output'
OUTPUT_CSV = 'resnet_features.csv'

weights = ResNet50_Weights.IMAGENET1K_V2

resnet = resnet50(weights=weights)

feature_extractor = nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

all_images = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

random.seed(42)

selected_images = random.sample(all_images, len(all_images) // 2)

rows = []

for filename in selected_images:

    image_path = os.path.join(
        INPUT_DIR,
        filename
    )

    print(f'Extracting features: {filename}')

    image = Image.open(image_path).convert('RGB')

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        features = feature_extractor(image_tensor)


    features = features.view(
        features.size(0),
        -1
    )

    features = features.numpy().flatten()

    row = {
        'image_name': filename
    }

    for i, value in enumerate(features):
        row[f'features{i}'] = value

    rows.append(row)

df = pd.DataFrame(rows)
print(df.shape)
print(df.head())

df.to_csv(OUTPUT_CSV, index=False)

print('\nFeature extraction completed.')
print(f'Saved to {OUTPUT_CSV}.')