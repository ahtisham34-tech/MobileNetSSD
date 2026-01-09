import cv2
import numpy as np
import pyttsx3
import time

# ===================== CONFIG =====================
CONF_THRESHOLD = 0.5
CAMERA_INDEX = 0   # 0 = laptop webcam | change if mobile webcam
ANNOUNCE_DELAY = 2 # seconds

# Known object widths in CM (approx)
KNOWN_WIDTHS = {
    "person": 50,
    "chair": 45,
    "bottle": 7
}

# ===================== LOAD MODEL =====================
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Essential classes for visually impaired assistance
IMPORTANT_CLASSES = {
    "person", "car", "bus", "motorbike", "bicycle", "chair", "sofa", "bottle"
}

# Add approximate widths (cm) for new important classes if missing
KNOWN_WIDTHS.update({
    "car": 180,
    "bus": 250,
    "motorbike": 80,
    "bicycle": 60,
    "sofa": 200,
    "bottle": 10 # Adjusted slightly
})

# ... (TTS setup remains) ...
engine = pyttsx3.init()
engine.setProperty("rate", 165)

last_announce_time = 0

# ===================== DISTANCE FUNCTIONS =====================
def calculate_focal_length(measured_distance, real_width, pixel_width):
    return (pixel_width * measured_distance) / real_width

def estimate_distance(real_width, focal_length, pixel_width):
    return (real_width * focal_length) / pixel_width

# ===================== SMOOTHING =====================
# Store history of distances for smoothing {label_id: [d1, d2, ...]}
history = {}
HISTORY_SIZE = 5

def get_smoothed_distance(obj_id, new_distance):
    if obj_id not in history:
        history[obj_id] = []
    
    history[obj_id].append(new_distance)
    if len(history[obj_id]) > HISTORY_SIZE:
        history[obj_id].pop(0)
        
    return sum(history[obj_id]) / len(history[obj_id])

# ===================== CAMERA =====================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[ERROR] Could not open camera (Index {CAMERA_INDEX}). Exiting...")
    exit()

time.sleep(2)

# ---- FOCAL LENGTH CALIBRATION (Person at 100cm) ----
KNOWN_DISTANCE = 100  # cm
FOCAL_LENGTH = None

print("[INFO] Calibrating focal length... Stand 1 meter away.")
print("[INFO] Collecting samples for calibration...")

calibration_samples = []
CALIBRATION_FRAMES = 20

while len(calibration_samples) < CALIBRATION_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Could not read frame from camera. Skipping calibration.")
        break

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    max_confidence = 0
    best_pixel_width = 0
    found_person = False

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.6:
            idx = int(detections[0,0,i,1])
            label = CLASSES[idx]

            if label == "person":
                # Find the detection with highest confidence in this frame
                if confidence > max_confidence:
                    max_confidence = confidence
                    
                    box = detections[0,0,i,3:7] * np.array(
                        [frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]]
                    )
                    (x1,y1,x2,y2) = box.astype("int")
                    best_pixel_width = x2 - x1
                    found_person = True

    if found_person and best_pixel_width > 0:
        f_len = calculate_focal_length(
            KNOWN_DISTANCE, KNOWN_WIDTHS["person"], best_pixel_width
        )
        calibration_samples.append(f_len)
        cv2.putText(frame, f"Samples: {len(calibration_samples)}/{CALIBRATION_FRAMES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Calibration - Stand 1m away", frame)
    if cv2.waitKey(1) == 27:
        break

try:
    cv2.destroyWindow("Calibration - Stand 1m away")
except cv2.error:
    pass

if len(calibration_samples) > 0:
    FOCAL_LENGTH = sum(calibration_samples) / len(calibration_samples)
    print(f"[OK] Calibrated Focal Length = {round(FOCAL_LENGTH, 2)}")
else:
    print("[WARNING] Focal length not calibrated. Using default value: 500")
    FOCAL_LENGTH = 500

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    announce_texts = []
    danger_zone = False # Flag if something is extremely close

    # We will iterate all detections to find valid ones
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > CONF_THRESHOLD:
            idx = int(detections[0,0,i,1])
            label = CLASSES[idx]

            # Filter: only process important classes
            if label not in IMPORTANT_CLASSES:
                continue
            
            # If we don't have width data, we can't estimate distance -> skip or provide generic warning?
            # For this code, we just skip to avoid crash/bad math.
            if label not in KNOWN_WIDTHS:
                continue

            box = detections[0,0,i,3:7] * np.array(
                [frame.shape[1], frame.shape[0],
                 frame.shape[1], frame.shape[0]]
            )
            (x1,y1,x2,y2) = box.astype("int")
            pixel_width = x2 - x1
            
            if pixel_width <= 0: continue

            distance = estimate_distance(
                KNOWN_WIDTHS[label], FOCAL_LENGTH, pixel_width
            )

            # --- DANGER ZONE CHECK ---
            if distance <= 50:
                danger_zone = True
                dist_str = f"{distance:.0f} cm"
                color = (0, 0, 255) # Red for danger
                label_text = f"STOP! {label} {dist_str}"
            else:
                color = (0, 255, 0) # Green for safe
                if distance <= 200:
                    dist_str = f"{distance:.0f} cm"
                else:
                    dist_str = f"{distance/100:.2f} m"
                label_text = f"{label} {dist_str}"

            # Add to announcement list
            # We add a "STOP" prefix to the speech if danger
            if distance <= 50:
                announce_texts.append(f"STOP {label} very close")
            else:
                announce_texts.append(f"{label} {dist_str}")

            cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---- SPEAKER ANNOUNCEMENT ----
    current_time = time.time()
    
    # Priority announcement for Danger Zone
    if danger_zone:
        # Announce immediately (or with very short delay) if danger
        if (current_time - last_announce_time) > 1.0: # Shorter delay for danger
            warning_msg = "STOP! Object detected extremely close."
            # Also read the specific objects causing danger
            engine.say(warning_msg)
            engine.runAndWait()
            last_announce_time = current_time
    else:
        # Standard announcement
        if announce_texts and (current_time - last_announce_time) > ANNOUNCE_DELAY:
            # Announce all detected objects
            # To avoid too long sentences, maybe limit to top 3 detections?
            # For now, joining all unique texts
            unique_texts = list(set(announce_texts))
            engine.say(", ".join(unique_texts))
            engine.runAndWait()
            last_announce_time = current_time

    cv2.imshow("MobileNet-SSD Assistive System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
