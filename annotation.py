import cv2
import os

# Paths for train and val
datasets = ["train", "val"]  

# Class names
class_names = [
    "articulated_truck", "bicycle", "bus", "car", "motorcycle",
    "non-motorized_vehicle", "pedestrian", "pickup_truck", "single_unit_truck", "work_van"
]

for dataset in datasets:
    image_folder = f"MIO-TCD-Localization/YOLO_dataset/{dataset}/images"
    label_folder = f"MIO-TCD-Localization/YOLO_dataset/{dataset}/labels"

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace(".jpg", ".txt"))

        # Read Image
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w, _ = img.shape

        # Read YOLO Label File
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    print(f"Debug: {data}")  # Print the contents of the line

                    if not data or data[0].lower() == 'nan':  # Check if the first value is missing or NaN
                        print("Skipping invalid label line:", line)
                        continue  # Skip this entry

                    try:
                        class_id = int(float(data[0]))  # Convert safely
                        x_center, y_center, width, height = map(float, data[1:])
                    except ValueError as e:
                        print(f"Could not parse line {line} - {e}")
                        continue

                    # Convert YOLO format to OpenCV format
                    x_min = int((x_center - width / 2) * w)
                    y_min = int((y_center - height / 2) * h)
                    x_max = int((x_center + width / 2) * w)
                    y_max = int((y_center + height / 2) * h)

                    # Draw Bounding Box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

                    # Put Class Label
                    label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overwrite the original image with annotated version
        cv2.imwrite(image_path, img)
        print(f"✅ Annotated: {image_path}")

print("All images in train and val folders have been annotated with bounding boxes!")
