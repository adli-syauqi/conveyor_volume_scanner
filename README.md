# Conveyor Volume Scanner

This project is a real-time computer vision script designed for OAK-D stereo cameras (specifically optimized for OAK-D Pro with IR laser support). It uses the DepthAI v3 API to stream synchronized RGB and depth feeds to estimate the volume of objects passing on a conveyor belt.

## Features
- Real-time Volume Calculation: Computes the volume (in cm³) of objects inside a predefined Region of Interest (ROI) by integrating depth pixel heights against a calibrated baseline.
- Dual-View Dashboard: Displays a side-by-side live feed of the RGB camera and a color-mapped depth visualization.
- Live Mask Visualization: Shows a binary mask of valid pixels currently being calculated as "load."
- Interactive Taring: Dynamically recalibrate the baseline conveyor belt distance on the fly.
- 3D Profiling (Optional): Includes commented-out Matplotlib integration to render a live 3D mesh surface of the ROI.
- IR Laser Support: Automatically attempts to enable the IR laser dot projector for better depth accuracy on flat/textureless surfaces.

## Hardware Requirements
- Camera: Luxonis OAK-D Pro.
- Connection: USB 3.0 cable providing sufficient power and bandwidth.

## Installation
If you are using Ubuntu, you need to setup udev rules first. Run the following command to grant permission to communicate with the Luxonis VPU via USB.
```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Ensure you have Python 3.7+ installed. You can install all necessary dependencies using the following one-line command. It is recommended to create a python virtual environment.
```
python3 -m pip install --upgrade pip
python3 -m venv depthai-env
source depthai-venv/bin/activate
pip install depthai opencv-python numpy matplotlib
```

## Usage
1. Connect your OAK-D camera to your computer.
2. Run the script:
    ```
    source depthai-env/bin/activate
    python conveyor_volume_scanner_v3.py
    ```
3. Focus the camera: Ensure the green/white bounding boxes are centered on the conveyor belt where objects will pass.

### Interactive Controls
- t - Tare Baseline: Press t when the conveyor belt is empty. This recalculates the BASELINE_MM based on the median depth inside the ROI, setting the current surface as the "zero" volume plane.
- q - Quit: Safely close the windows, shut down the DepthAI pipeline, and exit the script.

## Configuration & Calibration
To get accurate volume calculations, you must calibrate the physical constants at the top of the script according to your specific physical setup:
- RAW_BOX_SIZE: The pixel width/height of the Region of Interest (ROI) square.
- BASELINE_MM: The default distance from the camera lenses to the empty conveyor belt in millimeters. (Can be overridden at runtime using the t key).
- MM_PER_PIXEL: The physical size of a single pixel at the baseline depth. You must calibrate this by placing an object of known width/length on the belt and dividing its physical size in mm by its pixel size on the screen.
- HEIGHT_THRESHOLD: The minimum height (in mm) an object must be to be counted towards the volume. This acts as a noise filter to prevent small fluctuations on the belt surface from adding false volume.

## Troubleshooting
- No Volume Calculated: Ensure the BASELINE_MM is correct. If the depth reads further away than the baseline, the math assumes empty space. Press t to tare the empty belt.
- IR Laser Error Note: If you are using a standard OAK-D (non-Pro), the script will gracefully print a note that it cannot turn on the IR laser and will continue running normally.