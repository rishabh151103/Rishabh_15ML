import os
import pandas as pd
import cv2
import shutil
from sklearn.model_selection import train_test_split

csv_path = "MIO-TCD-Localization/gt_train.csv"
image_folder = "MIO-TCD-Localization/train"
output_base = "MIO-TCD-Localization/YOLO_dataset"
val_split = 0.2

for split in ['train', 'val']:
    for dir_type in ['images', 'labels']:
        os.makedirs(os.path.join(output_base, split, dir_type), exist_ok=True)

class_mapping = {
    "articulated_truck": 0, "bicycle": 1, "bus": 2, "car": 3, "motorcycle": 4,
    "non-motorized_vehicle": 5, "pedestrian": 6, "pickup_truck": 7, "single_unit_truck": 8, "work_van": 9
}
df = pd.read_csv(csv_path, delimiter=",", header=None, 
                 names=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"])

df["class_id"] = df["class_name"].map(class_mapping)
df.drop(columns=["class_name"], inplace=True)

df["image_id"] = df["image_id"].astype(str).str.zfill(8) 


unique_images = df['image_id'].unique()
train_images, val_images = train_test_split(unique_images, test_size=val_split, random_state=42)

split_dict = {img: 'train' for img in train_images}
split_dict.update({img: 'val' for img in val_images})

processed_images = set()

for image_id in unique_images:
    split = split_dict[image_id]

    extensions = [".jpg", ".png", ".jpeg"]
    print(f"Checking for {image_id} with extensions {extensions}...") 

    # Find correct image file
    image_path = None
    for ext in extensions:
        possible_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(possible_path):
            image_path = possible_path
            print(f"✅ Found: {image_path}")  # Debugging print
            break

    if image_path is None:
        print(f"Warning: No image found for {image_id} with extensions {extensions}")
        continue

    # Copy image
    if image_id not in processed_images:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read {image_path}")
            continue
            
        img_h, img_w, _ = img.shape
        dest_image_path = os.path.join(output_base, split, "images", f"{image_id}.jpg")
        shutil.copy2(image_path, dest_image_path)
        processed_images.add(image_id)
        print(f"Copied {image_id} to {dest_image_path}")

    # Convert BBox to YOLO format and save labels
    label_path = os.path.join(output_base, split, "labels", f"{image_id}.txt")
    with open(label_path, "w") as f:  # Overwrite to avoid duplicate entries
        for _, row in df[df["image_id"] == image_id].iterrows():
            class_id = row["class_id"]
            x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"Label saved: {label_path}")

#YAML Config
yaml_content = f"""
path: {output_base}
train: train/images
val: val/images
nc: {len(class_mapping)}
names: {list(class_mapping.keys())}
"""

with open(os.path.join(output_base, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

print(f"""
   Conversion completed! Dataset organized in: {output_base}
   Training images: {len(train_images)}
   Validation images: {len(val_images)}
   Total classes: {len(class_mapping)}
""")
