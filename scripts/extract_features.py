import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

INPUT_DIR = 'detect_output'
OUTPUT_DIR = 'extracted_features'

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(
        ('.jpg', '.jpeg', '.png')
    ):
        continue

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

    features_numpy = features.numpy()

    output_filename = (
        os.path.splitext(filename)[0]
        + '.npy'
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    np.save(output_path, features_numpy)

    print(f'Saved features: {output_path}')

print('\nFeature extraction completed.')