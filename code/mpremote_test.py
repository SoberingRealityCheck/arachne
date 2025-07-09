import subprocess
import time

from piprint import piprint

def ask_pico(file, function, *args):
    """
    Ask the pico to run a function on a file.
    """
    piprint(f"Asking Pico: {file}.{function}({args})")
    subprocess.run(["mpremote", "exec", f"import {file}; {file}.{function}(*{args})"])
    time.sleep(1)  # Allow some time for the command to execute

def main():
    """
    Main function to test the servo2040 module.
    """
    # Initialize servos
    ask_pico("servos", "initialize_servos")

    # Move servo 0 to 90 degrees
    ask_pico("servos", "move_servo", 0, 45)

    # Move all servos to min position
    ask_pico("servos", "servos_min")

    # Reset all servos to mid position
    ask_pico("servos", "servos_mid")

    # Move all servos to max position
    ask_pico("servos", "servos_max")

    # Disable all servos
    ask_pico("servos", "disable_servos")
    
if __name__ == "__main__":
    main()
