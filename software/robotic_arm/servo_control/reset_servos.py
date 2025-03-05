import pigpio
import time

pi = pigpio.pi()

# GPIO pin specifications for each Proto Vizzy V2 servo
SERVO_CLW = 4
SERVO_TOP = 17
SERVO_MID = 27
SERVO_BTM = 22

try:
    # Move all servos to default positions (90 degrees) with 1500 us pulse width
    pi.set_servo_pulsewidth(SERVO_CLW, 1500)
    time.sleep(1)
    pi.set_servo_pulsewidth(SERVO_TOP, 1500)
    time.sleep(1)
    pi.set_servo_pulsewidth(SERVO_MID, 1500)
    time.sleep(1)
    pi.set_servo_pulsewidth(SERVO_BTM, 1500)
    time.sleep(1)
    
    print("Reset Vizzy servos!")

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Stop all servos and clean up
    pi.set_servo_pulsewidth(SERVO_CLW, 0)  
    pi.set_servo_pulsewidth(SERVO_TOP, 0)  
    pi.set_servo_pulsewidth(SERVO_MID, 0)  
    pi.set_servo_pulsewidth(SERVO_BTM, 0) 
    pi.stop()  # Stop the pigpio instance
    print("Servos stopped and pigpio cleaned up.")
