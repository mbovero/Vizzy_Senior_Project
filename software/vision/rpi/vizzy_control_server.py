import pigpio
import socket
import json
import time
from threading import Thread, Event

from threading import Thread, Event

search_active = Event()

# GPIO pin specifications
SERVO_BTM = 22  # Bottom servo - horizontal/pan movements
SERVO_MID = 27
SERVO_TOP = 17  # Top servo - vertical/tilt movements

# Servo configuration
SERVO_MIN = 1000  # Minimum pulse width (us)
SERVO_MAX = 2000  # Maximum pulse width (us)
SERVO_CENTER = 1500  # Center position (us)

# Current positions
current_horizontal = SERVO_CENTER
current_vertical = SERVO_CENTER

def setup_servos(pi):
    """Initialize servos to center position"""
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)  # Middle servo
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)  # Bottom servo (horizontal)
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)  # Top servo (vertical)
    time.sleep(1)

def move_servos(pi, horizontal, vertical):
    """Move servos based on normalized inputs (-1 to 1)"""
    global current_horizontal, current_vertical
    
    # Calculate new positions
    horizontal_change = int(horizontal * 200)  # Adjust multiplier for sensitivity
    vertical_change = int(vertical * 200)
    
    new_horizontal = current_horizontal + horizontal_change  # Bottom servo
    new_vertical = current_vertical - vertical_change       # Top servo (inverted)
    
    # Constrain to valid range
    new_horizontal = max(SERVO_MIN, min(SERVO_MAX, new_horizontal))
    new_vertical = max(SERVO_MIN, min(SERVO_MAX, new_vertical))
    
    # Move servos
    pi.set_servo_pulsewidth(SERVO_BTM, new_horizontal)  # Bottom for horizontal
    pi.set_servo_pulsewidth(SERVO_TOP, new_vertical)    # Top for vertical
    
    # Update current positions
    current_horizontal = new_horizontal
    current_vertical = new_vertical

SEARCH_MIN = SERVO_MIN + 200
SEARCH_MAX = SERVO_MAX - 200
SEARCH_H_STEP = 250
SEARCH_V_STEP = 100
SEARCH_DELAY = 0.70

def search_pattern(pi):
    """Sweep servos through a predefined pattern while search_active is set."""
    global current_horizontal, current_vertical

    print("Search started")
    while search_active.is_set():
        for h in range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_H_STEP):
            if not search_active.is_set():
                break
            pi.set_servo_pulsewidth(SERVO_BTM, h)
            current_horizontal = h
            for v in range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_V_STEP):
                if not search_active.is_set():
                    break
                pi.set_servo_pulsewidth(SERVO_TOP, v)
                current_vertical = v
                time.sleep(SEARCH_DELAY)
        for h in range(SEARCH_MAX, SEARCH_MIN - 1, -SEARCH_H_STEP):
            if not search_active.is_set():
                break
            pi.set_servo_pulsewidth(SERVO_BTM, h)
            current_horizontal = h
            for v in range(SEARCH_MAX, SEARCH_MIN -1, -SEARCH_V_STEP):
                if not search_active.is_set():
                    break
                pi.set_servo_pulsewidth(SERVO_TOP, v)
                current_vertical = v
                time.sleep(SEARCH_DELAY)
    print("Search stopped")

def handle_client(conn, pi):
    search_thread = None
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            
            try:
                command = json.loads(data.decode('utf-8'))

                if command['type'] == 'move':
                    if not search_active.is_set():
                        move_servos(pi, command['horizontal'], command['vertical'])

                elif command['type'] == 'search':
                    if command['active']:
                        if not search_active.is_set():
                            search_active.set()
                            search_thread = Thread(target=search_pattern, args=(pi,))
                            search_thread.start()
                    else:
                        search_active.clear()
                        setup_servos(pi)  # Return to center

                elif command['type'] == 'stop':
                    search_active.clear()
                    break

            except json.JSONDecodeError:
                print("Invalid command received")
                
    finally:
        conn.close()
        search_active.clear()

def main():
    # Initialize pigpio
    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio daemon")
        return
    
    setup_servos(pi)
    
    # Set up TCP server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 65432))  # Listen on all interfaces
        s.listen()
        print("Server started, waiting for connections...")
        
        try:
            while True:
                conn, addr = s.accept()
                print(f"Connected by {addr}")
                Thread(target=handle_client, args=(conn, pi)).start()
                
        except KeyboardInterrupt:
            print("Server shutting down...")
            
        finally:
            # Clean up
            pi.set_servo_pulsewidth(SERVO_BTM, 0)
            pi.set_servo_pulsewidth(SERVO_TOP, 0)
            pi.stop()
            s.close()

if __name__ == "__main__":
    main()
