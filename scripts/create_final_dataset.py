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

features_df = pd.read_csv(
    "resnet_features.csv"
)

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
    .astype(str)
    .str.strip()
    .map(age_mapping)
)

final_df.to_csv(
    "final_dataset.csv",
    index=False
)

print(final_df.shape)

print(final_df[["image_name", "age", "age_class"]].head())

print("\nSaved to final_dataset.csv")