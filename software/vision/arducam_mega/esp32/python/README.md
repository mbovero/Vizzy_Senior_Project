# ArduCam Mega + ESP32 Serial Data Processing Scripts

This directory contains Python scripts for interfacing with an ArduCam Mega + ESP32 system over a serial connection. These scripts are used to log raw serial data, extract image data from the serial stream, and display real-time image streams.

## Python Scripts Overview

### 1. **raw_serial_logger.py**
   - **Purpose**: Logs raw serial data from MCU to a binary file.
   - **Functionality**:
     - Opens a serial connection.
     - Reads incoming serial data and writes it to a binary file.
     - Prints the received data in hexadecimal format for debugging.

### 2. **serial_to_JPG.py**
   - **Purpose**: Extracts image data from serial communication and saves it as a `.jpg` file.
   - **Functionality**:
     - Reads raw serial data and searches for a predefined header and footer.
     - Extracts the image data and determines its format.
     - Saves the extracted image as a `.jpg` file with a timestamp.
     - Runs continuously, processing incoming image data.

### 3. **serial_to_stream.py**
   - **Purpose**: Streams image data from serial communication in real-time using OpenCV.
   - **Functionality**:
     - Reads serial data and extracts image frames.
     - Decodes images using OpenCV and NumPy.
     - Continuously displays the latest image in an OpenCV window.
     - Runs in a separate thread to ensure smooth streaming.


## Python Environment & Dependencies

These scripts were developed using Python **3.11.11** and the following dependencies:

- `numpy`
- `opencv-python`
- `pyserial`

### Installing Dependencies
To install the required dependencies, run:
```sh
pip install -r requirements.txt
```


## Usage Instructions

1. **Setup the Serial Port**:
   - Modify the `port` parameter in each script to match your system’s serial port (e.g., `COM3` for Windows, `/dev/ttyUSB0` for Linux/Mac).
   
2. **Run the Desired Script**:
   - **Logging Raw Data**:
     ```sh
     python raw_serial_logger.py
     ```
   - **Extracting Images from Serial Data**:
     ```sh
     python serial_to_JPG.py
     ```
   - **Streaming Images in Real-Time**:
     ```sh
     python serial_to_stream.py
     ```

3. **Stop Execution**:
   - Press `Ctrl+C` to stop logging scripts.
   - Press `q` to close the OpenCV stream window.

---

For any issues, feel free to open a discussion or issue in the repository!

