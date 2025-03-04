import serial
import time

# Configure the serial port
ser = serial.Serial(
    port='COM5',       # Replace with your COM port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
    baudrate=115200,   # Match the baud rate in your Arduino code
    timeout=1          # Timeout for reading
)

# Open the serial connection
if not ser.is_open:
    ser.open()

# Function to log raw data
def log_raw_data():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"raw_data_{timestamp}.bin"
    with open(filename, "wb") as f:
        print(f"Logging raw data to {filename}...")
        while True:
            ser.reset_input_buffer()  # Flush the input buffer
            time.sleep(0.75)  # Add a small delay after resetting the microcontroller
            data = ser.read(ser.in_waiting or 1)
            if data:
                f.write(data)
                print(data.hex(), end=" ")  # Print raw data in hex format
            time.sleep(0.2)

# Main loop
try:
    log_raw_data()

except KeyboardInterrupt:
    print("Exiting...")

finally:
    ser.close()