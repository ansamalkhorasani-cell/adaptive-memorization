# -*- coding: utf-8 -*-
"""
DHT22 Noise Calibration Script

This script collects high-frequency samples from a DHT22 sensor in a completely stable,
insulated environment (e.g., inside a closed box or drawer in a room with no active AC/heating).
The data will be used to calculate the sensor's inherent noise floor (variance), which provides
scientific justification for the rolling variance filter threshold clamping.

Instructions:
1. Place the Raspberry Pi and DHT22 sensor in a stable, closed environment.
2. Adjust the PIN configuration below based on your hardware wiring.
3. Run the script for 2 to 3 hours:
   python dht22_noise_calibration.py
4. The script will save the results to 'dht22_noise_data.csv'.
"""

import time
import csv
import os

# --- CONFIGURATION ---
# Choose your preferred library by uncommenting the corresponding section.

# Option A: Using the newer Adafruit CircuitPython DHT library (Recommended)
try:
    import board
    import adafruit_dht
    # Use D4 for GPIO 4. Change board.D4 if you are using another pin.
    dht_device = adafruit_dht.DHT22(board.D4)
    def read_sensor():
        try:
            temp = dht_device.temperature
            hum = dht_device.humidity
            return temp, hum
        except RuntimeError as error:
            # DHT sensors are sensitive and can fail to read occasionally
            # print(error.args[0])
            return None, None
except ImportError:
    dht_device = None

# Option B: Using the older Adafruit_DHT library (Fallback)
if dht_device is None:
    try:
        import Adafruit_DHT
        SENSOR_TYPE = Adafruit_DHT.DHT22
        SENSOR_PIN = 4  # Change this to your GPIO pin number
        def read_sensor():
            hum, temp = Adafruit_DHT.read_retry(SENSOR_TYPE, SENSOR_PIN)
            return temp, hum
    except ImportError:
        # Dummy generator for testing/fallback if libraries are not installed yet
        import random
        print("[WARNING] No DHT library found. Using synthetic data generator for simulation.")
        def read_sensor():
            # Simulated base temp of 25C with minor noise and 0.1C resolution
            temp = round(25.0 + random.normalvariate(0, 0.05), 1)
            hum = round(50.0 + random.normalvariate(0, 0.1), 1)
            return temp, hum

# EXPERIMENT SETUP 
SAMPLING_INTERVAL = 10  # Seconds between readings
DURATION_HOURS = 3      # Total experiment duration
TOTAL_SAMPLES = int((DURATION_HOURS * 3600) / SAMPLING_INTERVAL)
CSV_FILE = "dht22_noise_data.csv"

print("="*60)
print("DHT22 NOISE CALIBRATION EXPERIMENT")
print("="*60)
print(f"Sampling every {SAMPLING_INTERVAL} seconds for {DURATION_HOURS} hours.")
print(f"Targeting {TOTAL_SAMPLES} samples.")
print(f"Data will be saved to: {os.path.abspath(CSV_FILE)}")
print("Ensure the sensor is in a sealed, thermally stable environment (e.g., closed box).")
print("Press Ctrl+C to terminate early.")
print("="*60)

# Create CSV header if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "sample_index", "temperature", "humidity"])

start_time = time.time()
sample_count = 0

try:
    while sample_count < TOTAL_SAMPLES:
        loop_start = time.time()
        
        temp, hum = read_sensor()
        
        if temp is not None and hum is not None:
            sample_count += 1
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            elapsed = time.time() - start_time
            
            # Save to CSV
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, sample_count, temp, hum])
            
            print(f"[{timestamp}] Sample {sample_count}/{TOTAL_SAMPLES} | Temp: {temp}°C | Hum: {hum}% | Elapsed: {elapsed/60:.1f} min")
        else:
            # In case of read failure, wait briefly and retry
            print("[INFO] Sensor read timeout/failure. Retrying in 2 seconds...")
            time.sleep(2)
            continue
            
        # Control exact interval
        elapsed_loop = time.time() - loop_start
        sleep_time = max(0.1, SAMPLING_INTERVAL - elapsed_loop)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n[INFO] Experiment terminated early by user.")

print(f"\n[DONE] Calibration complete. {sample_count} valid samples logged in '{CSV_FILE}'.")
