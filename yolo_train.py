from ultralytics import YOLO
import os
import glob
import multiprocessing

def get_last_checkpoint():
    """latest checkpoint"""
    checkpoints = glob.glob("runs/detect/train*/weights/last.pt")
    return max(checkpoints, key=os.path.getctime) if checkpoints else None

def main():
    last_checkpoint = get_last_checkpoint()

    if last_checkpoint:
        print(f"Resuming training from {last_checkpoint}")
        model = YOLO(last_checkpoint) 
        resume_training = True
    else:
        print("Starting fresh training with YOLOv8")
        model = YOLO("yolov8n.pt")
        resume_training = False

    model.train(
        data="MIO-TCD-Localization/YOLO_dataset/dataset.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device="cuda",
        save=True,
        resume=resume_training
    )

    metrics = model.val()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
