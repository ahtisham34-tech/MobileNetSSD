import cv2

print("Checking for cameras...")
found = False
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera found and working at index {i}")
            found = True
        else:
            print(f"Camera opened at index {i} but failed to read frame.")
        cap.release()
    else:
        print(f"No camera at index {i}")

if not found:
    print("No working camera found in indices 0-4.")
