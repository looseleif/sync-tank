import smbus2
import time

# I2C Setup
bus = smbus2.SMBus(1)
PCA9685_ADDRESS = 0x40

# Register addresses
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

# PCA9685 Initialization
bus.write_byte_data(PCA9685_ADDRESS, MODE1, 0x00)
time.sleep(0.005)

freq = 50
prescale_val = int(25000000.0 / (4096 * freq) - 1)
bus.write_byte_data(PCA9685_ADDRESS, MODE1, 0x10)
bus.write_byte_data(PCA9685_ADDRESS, PRESCALE, prescale_val)
bus.write_byte_data(PCA9685_ADDRESS, MODE1, 0x00)
time.sleep(0.005)
bus.write_byte_data(PCA9685_ADDRESS, MODE1, 0xA1)

# Set PWM Helper
def set_pwm(bank, on, off):
    reg_base = LED0_ON_L + 4 * bank
    bus.write_byte_data(PCA9685_ADDRESS, reg_base, on & 0xFF)
    bus.write_byte_data(PCA9685_ADDRESS, reg_base + 1, (on >> 8) & 0xFF)
    bus.write_byte_data(PCA9685_ADDRESS, reg_base + 2, off & 0xFF)
    bus.write_byte_data(PCA9685_ADDRESS, reg_base + 3, (off >> 8) & 0xFF)

# Motors: bank, midpoint, nudge, position, direction toggle
motors = [
    {"bank": 0, "midpoint": 400, "nudge": 130, "position": 400, "dir": 1},
    {"bank": 1, "midpoint": 0, "nudge": 400, "position": 400, "dir": 1},
    {"bank": 2, "midpoint": 450, "nudge": 100, "position": 500, "dir": 1}
]

step_size = 2      # How small each motion increment is (lower = smoother, higher = faster)
step_delay = 0.01  # Delay between steps (lower = faster)

# Initialize motors to midpoint
for motor in motors:
    set_pwm(motor["bank"], 0, motor["midpoint"])
    motor["position"] = motor["midpoint"]

print("\nSmooth Joint Control Demo")
print("Press 1, 2, or 3 to nudge joints 0, 1, or 2 alternately forward/backward gradually")
print("Type q to quit\n")

try:
    while True:
        cmd = input("> ").strip().lower()

        if cmd == "q":
            print("Exiting...")
            break

        if cmd not in ["1", "2", "3"]:
            print("Invalid input. Use 1, 2, 3 or q to quit.")
            continue

        motor_idx = int(cmd) - 1
        if motor_idx >= len(motors):
            print(f"No motor for input {cmd}")
            continue

        motor = motors[motor_idx]
        direction = motor["dir"]

        target = motor["position"] + motor["nudge"] * direction

        # Clamp target position
        lower_limit = motor["midpoint"] - motor["nudge"] * 5
        upper_limit = motor["midpoint"] + motor["nudge"] * 5
        target = max(lower_limit, min(upper_limit, target))

        # Smoothly move to target
        while abs(motor["position"] - target) >= step_size:
            if motor["position"] < target:
                motor["position"] += step_size
            else:
                motor["position"] -= step_size

            set_pwm(motor["bank"], 0, int(motor["position"]))
            time.sleep(step_delay)

        motor["position"] = target
        set_pwm(motor["bank"], 0, int(motor["position"]))

        print(f"Motor {motor_idx} moved to {motor['position']:.1f}")

        # Reverse direction next press
        motor["dir"] *= -1

except KeyboardInterrupt:
    print("\nStopped by user")