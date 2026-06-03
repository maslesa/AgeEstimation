import pandas as pd

metadata_files = [
    'dataset/fold_0_data.txt',
    'dataset/fold_1_data.txt',
    'dataset/fold_2_data.txt',
    'dataset/fold_3_data.txt',
    'dataset/fold_4_data.txt'
]

metadata_frames = []

for file in metadata_files:
    df = pd.read_csv(file, sep='\t')
    metadata_frames.append(df)

metadata = pd.concat(metadata_frames, ignore_index=True)

metadata["image_name"] = (
    metadata["user_id"]
    + "_coarse_tilt_aligned_face."
    + metadata["face_id"].astype(str)
    + "."
    + metadata["original_image"]
)

valid_age_classes = [
    "(0, 2)",
    "(4, 6)",
    "(8, 13)",
    "(15, 20)",
    "(25, 32)",
    "(38, 43)",
    "(48, 53)",
    "(60, 100)"
]

metadata["age"] = (
    metadata["age"]
    .astype(str)
    .str.strip()
)

metadata = metadata[
    metadata["age"].isin(valid_age_classes)
]

print(f'Rows after filtering age classes: {len(metadata)}')


features_df = pd.read_csv("resnet_features.csv")

print(f'Feature rows: {len(features_df)}')

final_df = features_df.merge(
    metadata[
        [
            "image_name",
            "age",
            "gender"
        ]
    ],
    on="image_name",
    how="inner"
)

print(f'Rows after merge: {len(final_df)}')

age_mapping = {
    "(0, 2)": 0,
    "(4, 6)": 1,
    "(8, 13)": 2,
    "(15, 20)": 3,
    "(25, 32)": 4,
    "(38, 43)": 5,
    "(48, 53)": 6,
    "(60, 100)": 7
}

final_df["age_class"] = (
    final_df["age"]
    .map(age_mapping)
)

evaluation_mapping = {
    0: 0,  # (0,2)
    1: 0,  # (4,6)
    2: 0,  # (8,13)

    3: 1,  # (15,20)

    4: 2,  # (25,32)
    5: 2,  # (38,43)
    6: 2,  # (48,53)
    7: 2   # (60,100)
}

final_df["evaluation_class"] = (
    final_df["age_class"]
    .map(evaluation_mapping)
)

final_df.to_csv("final_dataset.csv", index=False)
print("\nDataset saved successfully!")

print("\nFinal shape:")
print(final_df.shape)

print("\nAge class distribution:")
print(
    final_df["age_class"]
    .value_counts()
    .sort_index()
)

print("\nEvaluation class distribution:")
print(
    final_df["evaluation_class"]
    .value_counts()
    .sort_index()
)

print("\nPreview:")
print(
    final_df[
        [
            "image_name",
            "age",
            "age_class",
            "evaluation_class"
        ]
    ].head()
)