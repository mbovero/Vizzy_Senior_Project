import serial
import time
import cv2
import numpy as np
import threading

# Configure the serial port
ser = serial.Serial(
    port='COMX',       # Replace with your COM port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
    baudrate=115200,   # Match the baud rate in your Arduino code
    timeout=1          # Timeout for reading
)

# Open the serial connection
if not ser.is_open:
    ser.open()

# Constants for the custom protocol
HEADER = bytes([0xFF, 0xAA, 0x01])  # Custom header indicating start of image data
FOOTER = bytes([0xFF, 0xBB])        # Custom footer indicating end of image data

# Global variable for storing the latest image
latest_image = None

# Function to continuously read and process image data
def process_serial_data():
    global latest_image

    buffer = bytearray()  # Buffer to store incoming data

    while True:
        try:
            data = ser.read(ser.in_waiting or 1)
            if data:
                buffer.extend(data)  # Add new data to the buffer
            
            # Look for header
            header_index = buffer.find(HEADER)
            if header_index == -1:
                continue

            # Check if we have enough data for the length field
            if len(buffer) < header_index + 7:
                continue

            # Extract image length (4 bytes after header)
            image_length = int.from_bytes(buffer[header_index + 3:header_index + 7], byteorder='little')

            # Check if we have enough data for the full image and footer
            if len(buffer) < header_index + 8 + image_length + 2:
                continue

            # Extract the image data
            image_data = buffer[header_index + 8:header_index + 8 + image_length]

            # Check if footer is valid
            footer_data = buffer[header_index + 8 + image_length:header_index + 8 + image_length + 2]
            if footer_data != FOOTER:
                print("Invalid footer. Skipping corrupted data...")
                buffer = buffer[header_index + 1:]  # Discard up to header+1 and retry
                continue

            # Convert the image data to a NumPy array
            image_array = np.frombuffer(image_data, dtype=np.uint8)

            # Decode the image (assuming JPG format)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is not None:
                latest_image = img  # Store the latest image for display
            
            # Remove processed data from the buffer
            buffer = buffer[header_index + 8 + image_length + 2:]

        except Exception as e:
            print(f"Error processing data: {e}")

# Function to display the latest image
def display_stream():
    while True:
        if latest_image is not None:
            cv2.imshow("ArduCam Stream", latest_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Start serial data processing in a separate thread
serial_thread = threading.Thread(target=process_serial_data, daemon=True)
serial_thread.start()

# Start OpenCV image display
display_stream()
