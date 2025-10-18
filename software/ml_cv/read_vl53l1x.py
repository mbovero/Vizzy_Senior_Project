import time
import board, busio
import adafruit_vl53l1x

# Initialize I²C and sensor
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_vl53l1x.VL53L1X(i2c)

sensor.start_ranging()          # begin continuous measurements

try:
    while True:
        if sensor.data_ready:
            dist = sensor.distance   # distance in millimeters
            sensor.clear_interrupt()
            print(f"Distance: {dist} mm")
            time.sleep(0.05)         # 20 Hz print rate
finally:
    sensor.stop_ranging()
