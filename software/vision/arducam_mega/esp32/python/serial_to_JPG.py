import serial
import time
import os

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

# File to store raw serial data
RAW_DATA_FILE = "raw_data.bin"

# Function to initialize the raw data file
def initialize_raw_data_file():
    if os.path.exists(RAW_DATA_FILE):
        print(f"Clearing existing file: {RAW_DATA_FILE}")
        os.remove(RAW_DATA_FILE)  # Delete the file if it exists
    # Create an empty file
    with open(RAW_DATA_FILE, "wb") as f:
        pass  # Just create the file
    print(f"Initialized empty file: {RAW_DATA_FILE}")

# Function to log raw data
def log_raw_data():
    print("Logging raw data to file...")
    with open(RAW_DATA_FILE, "ab") as f:  # Open in append mode
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                f.write(data)
            time.sleep(0.1)

# Function to parse raw data and extract images
def parse_raw_data():
    print("Parsing raw data for images...")
    while True:
        try:
            with open(RAW_DATA_FILE, "rb") as f:
                raw_data = f.read()

            # Search for the header in the raw data
            header_index = raw_data.find(HEADER)
            if header_index == -1:
                print("No valid header found. Waiting for more data...")
                time.sleep(1)
                continue

            # Extract the image length (4 bytes after the header)
            if len(raw_data) < header_index + 7:  # Header (3) + Length (4)
                print("Incomplete data. Waiting for more data...")
                time.sleep(1)
                continue

            image_length = int.from_bytes(raw_data[header_index + 3:header_index + 7], byteorder='little')
            print(f"Image length: {image_length} bytes")

            # Extract the image format (1 byte after the length)
            if len(raw_data) < header_index + 8:  # Header (3) + Length (4) + Format (1)
                print("Incomplete data. Waiting for more data...")
                time.sleep(1)
                continue

            image_format = raw_data[header_index + 7]
            print(f"Image format: {image_format:02X}")

            # Check if the full image data is available
            if len(raw_data) < header_index + 8 + image_length + 2:  # Header (3) + Length (4) + Format (1) + Image + Footer (2)
                print("Incomplete image data. Waiting for more data...")
                time.sleep(1)
                continue

            # Extract the image data and footer
            image_data = raw_data[header_index + 8:header_index + 8 + image_length]
            footer_data = raw_data[header_index + 8 + image_length:header_index + 8 + image_length + 2]

            # Verify the footer
            if footer_data != FOOTER:
                print("Invalid footer. Discarding data...")
                # Discard data up to the invalid footer
                with open(RAW_DATA_FILE, "wb") as f:
                    f.write(raw_data[header_index + 8 + image_length + 2:])
                continue

            # Save the image data to a file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.jpg"
            with open(filename, "wb") as f:
                f.write(image_data)
            print(f"Image saved as {filename}")

            # Discard the processed data from the raw data file
            with open(RAW_DATA_FILE, "wb") as f:
                f.write(raw_data[header_index + 8 + image_length + 2:])

        except Exception as e:
            print(f"Error parsing raw data: {e}")
            time.sleep(1)

# Main loop
try:
    # Initialize the raw data file
    initialize_raw_data_file()

    # Start logging raw data in a separate thread
    import threading
    logging_thread = threading.Thread(target=log_raw_data, daemon=True)
    logging_thread.start()

    # Start parsing raw data
    parse_raw_data()

except KeyboardInterrupt:
    print("Exiting...")

finally:
    ser.close()