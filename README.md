# MobileNetSSD Object Detection & Distance Estimation

This project is a real-time object detection and distance estimation system designed to assist visually impaired users. It uses a **MobileNetSSD** deep learning model to detect objects via a webcam and provides audio feedback using Text-to-Speech (TTS).

## Features

- **Real-Time Object Detection**: Identifies common objects such as people, cars, chairs, bottles, and more using a pre-trained MobileNetSSD model.
- **Distance Estimation**: Calculates the approximate distance of objects using monocular vision principles (triangular similarity).
- **Audio Feedback**: Announces detected objects and their distances to the user.
- **Safety Alerts**:
  - **Danger Zone (< 50cm)**: Triggers an immediate "STOP" warning.
  - **Proximity Alerts**: Reports distances in centimeters for closer objects (< 2m) and meters for further ones.
- **Auto-Calibration**: utility to calibrate the camera's focal length by standing 1 meter away at startup.

## Prerequisites

Ensure you have Python installed (Python 3.7+ recommended). You will need the following libraries:

```bash
pip install opencv-python numpy pyttsx3
```

## Files Description

- **`main.py`**: The main application script that runs the detection loop, distance calculation, and audio feedback.
- **`check_camera.py`**: A utility script to check available camera indices if the default webcam doesn't load.
- **`MobileNetSSD_deploy.prototxt`**: The definition of the neural network architecture.
- **`MobileNetSSD_deploy.caffemodel`**: The pre-trained weights for the model.

## Usage

1. **Clone or Download** this repository.
2. **Install Dependencies** as listed above.
3. **Run the Application**:
   ```bash
   python main.py
   ```
4. **Calibration**:
   - When the app starts, it will enter a calibration mode.
   - **Stand exactly 1 meter (100 cm)** away from the camera.
   - The system will detect a "person" and automatically calculate the focal length.
   - Once calibrated, the main detection loop begins.
5. **Operation**:
   - The system will continuously scan for objects.
   - If an object is detected, it will be highlighted with a bounding box and labeled with the distance.
   - The system will speak out the object's name and distance.
6. **Exit**:
   - Press `q` to quit the application.

## Configuration

You can adjust settings in the `CONFIG` section of `main.py`:

- `CONF_THRESHOLD`: Minimum confidence for a detection to be considered valid (default `0.5`).
- `CAMERA_INDEX`: The camera ID to use (default `0`). Use `check_camera.py` to find other IDs if needed.
- `ANNOUNCE_DELAY`: Time in seconds between standard audio announcements to prevent spamming.

## Troubleshooting

- **Camera not found**: Run `python check_camera.py` to see which camera index is active, then update `CAMERA_INDEX` in `main.py`.
- **Audio issues**: Ensure your system volume is up and that `pyttsx3` drivers are correctly installed for your OS.

## Credits

- Uses the MobileNet-SSD model trained on the COCO dataset.
- Built with OpenCV and Python.
