import os
import shutil

SOURCE_DIR = 'dataset/faces'
DESTINATION_DIR = 'prepared_dataset'

os.makedirs(DESTINATION_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (
    '.jpg',
    '.jpeg',
    '.png'
)

copied_images = 0

for root, dirs, files in os.walk(SOURCE_DIR):

    parent_folder = os.path.basename(root)

    for file in files:

        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        source_path = os.path.join(root, file)

        new_filename = f'{parent_folder}_{file}'

        destination_path = os.path.join(
            DESTINATION_DIR,
            new_filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        copied_images += 1

        print(f'Copied: {new_filename}')


print(f'Total copied images: {copied_images}')
print('Dataset preparation completed.')
