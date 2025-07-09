from servo import Servo, servo2040
import time

"""
This module establishes the interface by which the servos can be controlled by external scripts.
It provides functions to initialize, move, and reset multiple servos connected to the servo2040 board.


==========================
Functions:
==========================

initialize_servos() - Enables all servos and puts them in their mid position.

move_servo(servo, angle) - Moves the specified servo to the given angle.

servos_min() - Moves all servos to their minimum position.

servos_mid() - Resets all servos to their mid position.

servos_max() - Moves all servos to their maximum position.

disable_servos() - Disables all servos.
"""


# Create a list of servos for pins 0 to 3. Up to 16 servos can be created
START_PIN = servo2040.SERVO_1
END_PIN = servo2040.SERVO_3
servos = [Servo(i) for i in range(START_PIN, END_PIN + 1)]

def initialize_servos():
    for s in servos:
        s.enable()
        time.sleep(0.05)

def move_servo(servo, angle):
    # Move the specified servo to the given angle
    s = servos[servo]
    print(f"Moving servo on pin {s} to angle {angle}")
    s.value(angle)
    time.sleep(.05)

def servos_min():
    # Move all servos to their minimum position
    print("Moving all servos to min position.")
    for s in servos:
        s.to_min()
        time.sleep(.05)

def servos_mid():
    # Reset all servos to their mid position
    print("Resetting all servos to mid position.")
    for s in servos:
        s.to_mid()
        time.sleep(.05)
        
def servos_max():
    # Move all servos to their maximum position
    print("Moving all servos to max position.")
    for s in servos:
        s.to_max()
        time.sleep(.05)

def disable_servos():
    # Disable all servos
    print("Disabling all servos.")
    for s in servos:
        s.disable()
        time.sleep(.05)



if __name__ == "__main__":
    # Example usage
    initialize_servos()  # Enable all servos
    move_servo(0, 45)  # Move servo 0 to 45 degrees
    time.sleep(1)
    servos_mid()     # Reset all servos to mid position
    time.sleep(1)
    move_servo(1, 90)  # Move servo 1 to 90 degrees
    time.sleep(1)
    reset_servos()     # Reset all servos to mid position