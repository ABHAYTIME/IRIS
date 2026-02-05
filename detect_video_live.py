import cv2
import time
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
WEIGHTS = "m1.pt"      # your trained model file in same folder
VIDEO   = "test1.mp4"     # put your video name here
CONF_THRES = 0.05        # lower if it's not detecting (try 0.03)
IMG_SIZE = 640
# -----------------------------------------

model = YOLO(WEIGHTS)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO}")

frame_id = 0
prev_t = time.time()

# Optional: video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # YOLO prediction
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]

    # Draw boxes (Ultralytics built-in)
    annotated = results.plot()

    # --- Frame info ---
    now = time.time()
    fps = 1.0 / (now - prev_t) if now != prev_t else 0
    prev_t = now

    num_det = 0 if results.boxes is None else len(results.boxes)

    # Best confidence among detections (if any)
    best_conf = 0.0
    if num_det > 0:
        best_conf = float(results.boxes.conf.max().cpu().numpy())

    info1 = f"Frame: {frame_id}   FPS: {fps:.1f}"
    info2 = f"Detections: {num_det}   BestConf: {best_conf:.3f}   ConfThres: {CONF_THRES}"

    cv2.putText(annotated, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(annotated, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("IRIS Accident Detection (Live)", annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
