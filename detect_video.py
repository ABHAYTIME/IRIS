from ultralytics import YOLO

MODEL_PATH = "m1.pt"
VIDEO_PATH = "test1.mp4"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=VIDEO_PATH,
    conf=0.25,     # try 0.15 if missing detections
    iou=0.5,
    imgsz=960,
    save=True      # saves annotated video
)

print("Done. Check runs/detect/ for the output video.")
