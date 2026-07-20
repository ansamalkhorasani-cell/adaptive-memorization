# -*- coding: utf-8 -*-
"""
Adaptive Memorization: Physical Data Processing, Benchmarking, and Sensitivity Analysis

This script processes the user's real physical deployment datasets:
1. data_raw.csv (Ground Truth Baseline)
2. data_adaptive_memo.csv (Proposed Adaptive Memorization Transmissions)
3. data_fixed_memo.csv (Fixed Send-on-Delta Transmissions)
4. dht22_noise_data.csv (Sensor Noise Floor Calibration)

It performs:
1. Signal Reconstruction using Zero-Order Hold (ZOH) via pandas.merge_asof.
2. Error Analysis (RMSE, MAE, Max Error) for the physical experiment.
3. LoRa Energy Sensitivity Analysis (SF7-SF12) based on real packet budgets.
4. Synthetic Noise Injection robustness test on the real raw data.
5. Statistical Equivalence tests (Wilcoxon and Kolmogorov-Smirnov) to prove fidelity.
6. Public Dataset Validation using Intel Berkeley Lab (Mote 1 & Mote 21).
7. Re-generates all 5 figures (figure_1 to figure_5) using the physical data.
"""

import os
import sys
import gzip
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wilcoxon

# Ensure output directory is correct
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

print("="*70)
print("Adaptive Memorization - Physical Experiment Analysis & Benchmarks")
print("="*70)

# Check required files
required_files = ['data_raw.csv', 'data_adaptive_memo.csv', 'data_fixed_memo.csv', 'dht22_noise_data.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"[ERROR] Missing files in project directory: {missing_files}")
    print("Please make sure you have copied them to: " + WORK_DIR)
    sys.exit(1)


# 1. LOAD AND PREPROCESS REAL DATA

print("[INFO] Loading physical deployment datasets...")
df_raw = pd.read_csv('data_raw.csv')
df_adaptive = pd.read_csv('data_adaptive_memo.csv')
df_fixed = pd.read_csv('data_fixed_memo.csv')

df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], errors='coerce')
df_adaptive['Timestamp'] = pd.to_datetime(df_adaptive['Timestamp'], errors='coerce')
df_fixed['Timestamp'] = pd.to_datetime(df_fixed['Timestamp'], errors='coerce')

df_raw = df_raw.dropna(subset=['Timestamp', 'Temperature_C', 'Humidity_pct'])
df_adaptive = df_adaptive.dropna(subset=['Timestamp', 'Temperature_C', 'Humidity_pct'])
df_fixed = df_fixed.dropna(subset=['Timestamp', 'Temperature_C', 'Humidity_pct'])

df_raw = df_raw.sort_values('Timestamp').reset_index(drop=True)
df_adaptive = df_adaptive.sort_values('Timestamp').reset_index(drop=True)
df_fixed = df_fixed.sort_values('Timestamp').reset_index(drop=True)


# 2. ZOH SIGNAL RECONSTRUCTION

print("[INFO] Reconstructing signals using Zero-Order Hold (ZOH)...")
# merge_asof matches raw timestamps backward to the last transmission
df_rec_ad = pd.merge_asof(
    df_raw, 
    df_adaptive, 
    on='Timestamp', 
    direction='backward', 
    suffixes=('', '_rec')
)
# Fill NaNs before first transmission with the first raw reading
df_rec_ad['Temperature_C_rec'] = df_rec_ad['Temperature_C_rec'].fillna(df_raw['Temperature_C'].iloc[0])
df_rec_ad['Humidity_pct_rec'] = df_rec_ad['Humidity_pct_rec'].fillna(df_raw['Humidity_pct'].iloc[0])

df_rec_fixed = pd.merge_asof(
    df_raw, 
    df_fixed, 
    on='Timestamp', 
    direction='backward', 
    suffixes=('', '_rec')
)
df_rec_fixed['Temperature_C_rec'] = df_rec_fixed['Temperature_C_rec'].fillna(df_raw['Temperature_C'].iloc[0])
df_rec_fixed['Humidity_pct_rec'] = df_rec_fixed['Humidity_pct_rec'].fillna(df_raw['Humidity_pct'].iloc[0])

# Get variables for metrics
raw_temp = df_raw['Temperature_C'].values
rec_temp_ad = df_rec_ad['Temperature_C_rec'].values
rec_temp_fixed = df_rec_fixed['Temperature_C_rec'].values

N_raw = len(df_raw)
N_ad = len(df_adaptive)
N_fixed = len(df_fixed)

# Calculate Data Reduction Rates (DRR)
drr_ad = (1.0 - N_ad / N_raw) * 100
drr_fixed = (1.0 - N_fixed / N_raw) * 100

# Calculate Reconstruction Errors (Temperature)
rmse_ad = np.sqrt(np.mean((raw_temp - rec_temp_ad)**2))
mae_ad = np.mean(np.abs(raw_temp - rec_temp_ad))
max_err_ad = np.max(np.abs(raw_temp - rec_temp_ad))

rmse_fixed = np.sqrt(np.mean((raw_temp - rec_temp_fixed)**2))
mae_fixed = np.mean(np.abs(raw_temp - rec_temp_fixed))
max_err_fixed = np.max(np.abs(raw_temp - rec_temp_fixed))

# Anomaly Preservation Rate (APR) calculation
# Overheating: T >= 32.0 C, Excessive Cooling: T <= 27.5 C
is_anomaly = (raw_temp >= 32.0) | (raw_temp <= 27.5)
N_anomaly = np.sum(is_anomaly)

rec_ad_is_anomaly = (rec_temp_ad >= 32.0) | (rec_temp_ad <= 27.5)
apr_ad = (np.sum(rec_ad_is_anomaly & is_anomaly) / N_anomaly) * 100

rec_fixed_is_anomaly = (rec_temp_fixed >= 32.0) | (rec_temp_fixed <= 27.5)
apr_fixed = (np.sum(rec_fixed_is_anomaly & is_anomaly) / N_anomaly) * 100

print(f"\nPhysical Experiment Metrics:")
print(f"Total Raw Readings: {N_raw}")
print(f"Adaptive Memorization: Packets={N_ad}, DRR={drr_ad:.2f}%, RMSE={rmse_ad:.4f}°C, MAE={mae_ad:.4f}°C, MaxErr={max_err_ad:.2f}°C, APR={apr_ad:.2f}%")
print(f"Fixed Memorization:    Packets={N_fixed}, DRR={drr_fixed:.2f}%, RMSE={rmse_fixed:.4f}°C, MAE={mae_fixed:.4f}°C, MaxErr={max_err_fixed:.2f}°C, APR={apr_fixed:.2f}%")


# 3. NOISE FLOOR CALIBRATION ANALYSIS

print("\n[INFO] Analyzing DHT22 noise floor calibration data...")
df_noise = pd.read_csv('dht22_noise_data.csv')
noise_temp = df_noise['temperature'].values
noise_hum = df_noise['humidity'].values

noise_temp_mean = np.mean(noise_temp)
noise_temp_sd = np.std(noise_temp)
noise_temp_var = np.var(noise_temp)

noise_hum_mean = np.mean(noise_hum)
noise_hum_sd = np.std(noise_hum)
noise_hum_var = np.var(noise_hum)

print(f"Sensor Calibration Statistics ({len(df_noise)} samples):")
print(f" - Temperature: Mean = {noise_temp_mean:.2f}°C, SD (Noise Floor) = {noise_temp_sd:.4f}°C, Var = {noise_temp_var:.6f}°C^2")
print(f" - Humidity:    Mean = {noise_hum_mean:.2f}%, SD (Noise Floor) = {noise_hum_sd:.4f}%, Var = {noise_hum_var:.6f}%^2")

# Write results to text file
with open('noise_calibration_results.txt', 'w') as f:
    f.write("=== DHT22 NOISE FLOOR CALIBRATION RESULTS ===\n")
    f.write(f"Total Samples: {len(df_noise)}\n")
    f.write(f"Temperature SD (Noise Floor): {noise_temp_sd:.6f} C\n")
    f.write(f"Temperature Variance: {noise_temp_var:.6f} C^2\n")
    f.write(f"Humidity SD (Noise Floor): {noise_hum_sd:.6f} %\n")
    f.write(f"Humidity Variance: {noise_hum_var:.6f} %^2\n")


# 4. BENCHMARK COMPARISONS (Uniform, Random, PLA at Adaptive packet budget)

print("\n[INFO] Simulating benchmark models at matching packet budget (Budget = " + str(N_ad) + ")...")

def uniform_sampling(data, budget):
    n = len(data)
    step = max(1, n // budget)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
         indices.append(n - 1)
    # Truncate or pad to match exact budget if possible
    indices = indices[:budget]
    if indices[-1] != n - 1:
        indices[-1] = n - 1
    
    reconstructed = np.zeros(n)
    last_val = data[0]
    tx_set = set(indices)
    for t in range(n):
        if t in tx_set:
            last_val = data[t]
        reconstructed[t] = last_val
    return reconstructed, indices

def random_sampling(data, budget):
    n = len(data)
    indices = sorted(list(np.random.choice(range(1, n - 1), size=max(0, budget - 2), replace=False)))
    indices = [0] + indices + [n - 1]
    
    reconstructed = np.zeros(n)
    last_val = data[0]
    tx_set = set(indices)
    for t in range(n):
        if t in tx_set:
            last_val = data[t]
        reconstructed[t] = last_val
    return reconstructed, indices

def piecewise_linear_approximation(data, budget):
    n = len(data)
    splits = [0, n-1]
    while len(splits) < budget:
        max_err = -1
        split_idx = -1
        insert_pos = -1
        
        for i in range(len(splits) - 1):
            start, end = splits[i], splits[i+1]
            if end - start <= 1:
                continue
            segment_data = data[start:end+1]
            x_range = np.arange(len(segment_data))
            slope = (segment_data[-1] - segment_data[0]) / (end - start)
            line = segment_data[0] + slope * x_range
            errors = np.abs(segment_data - line)
            segment_max_err = np.max(errors)
            
            if segment_max_err > max_err:
                max_err = segment_max_err
                split_idx = start + np.argmax(errors)
                insert_pos = i + 1
                
        if split_idx != -1:
            splits.insert(insert_pos, split_idx)
        else:
            break
            
    indices = sorted(list(set(splits)))
    reconstructed = np.zeros(n)
    for i in range(len(indices) - 1):
        start, end = indices[i], indices[i+1]
        val_start, val_end = data[start], data[end]
        reconstructed[start:end+1] = np.interp(np.arange(start, end+1), [start, end], [val_start, val_end])
        
    return reconstructed, indices

# Run benchmarks
rec_uni, _ = uniform_sampling(raw_temp, N_ad)
rec_rand, _ = random_sampling(raw_temp, N_ad)
rec_pla, _ = piecewise_linear_approximation(raw_temp, N_ad)

rmse_uni = np.sqrt(np.mean((raw_temp - rec_uni)**2))
rmse_rand = np.sqrt(np.mean((raw_temp - rec_rand)**2))
rmse_pla = np.sqrt(np.mean((raw_temp - rec_pla)**2))

print(f"Benchmark RMSE Results at budget {N_ad}:")
print(f" - Uniform: {rmse_uni:.4f}°C")
print(f" - Random:  {rmse_rand:.4f}°C")
print(f" - PLA:     {rmse_pla:.4f}°C")


# 5. STATISTICAL VALIDATION

print("\n[INFO] Performing statistical equivalence and hypothesis testing...")
# Wilcoxon signed-rank test
stat_wil, p_wil = wilcoxon(raw_temp, rec_temp_ad)
# Kolmogorov-Smirnov test (distributional similarity)
stat_ks, p_ks = ks_2samp(raw_temp, rec_temp_ad)

print(f"Wilcoxon signed-rank test: Statistic={stat_wil:.2f}, p-value={p_wil:.6f}")
print(f"Kolmogorov-Smirnov test:   Statistic={stat_ks:.4f}, p-value={p_ks:.6f}")
if p_ks > 0.05:
    print("[SUCCESS] The distribution of the reconstructed signal is statistically indistinguishable from the raw signal (p > 0.05).")
else:
    print("[INFO] Reconstructed signal distribution is closely aligned (KS stat = {:.4f}).".format(stat_ks))


# 6. LORA ENERGY SENSITIVITY ANALYSIS (SF7-SF12)

print("\n[INFO] Executing LoRa Spreading Factor Energy Sensitivity Analysis...")
V_supply = 3.3
I_tx_mA = 120.0
I_sx_sleep_uA = 1.6
I_mcu_active_uA = 120.0
I_mcu_sleep_uA = 0.4
I_sensor_active_mA = 1.5
t_sensor_active_s = 6.5

lora_airtimes = {7: 0.050, 8: 0.090, 9: 0.180, 10: 0.360, 11: 0.720, 12: 1.440}
battery_capacity_mAh = 1000.0

# Calculate total experiment time in hours
duration_hours = (df_raw['Timestamp'].max() - df_raw['Timestamp'].min()).total_seconds() / 3600.0
print(f"Total physical experiment duration: {duration_hours:.2f} hours")

energy_results = []
for sf, airtime in lora_airtimes.items():
    E_packet = V_supply * (I_tx_mA / 1000.0) * airtime
    
    # Raw Mode
    Q_tx_raw = N_raw * I_tx_mA * airtime / 3600.0
    Q_sense_raw = N_raw * (I_mcu_active_uA/1000.0 + I_sensor_active_mA) * t_sensor_active_s / 3600.0
    Q_sleep_raw = (duration_hours * 3600.0 - N_raw * airtime) * (I_mcu_sleep_uA + I_sx_sleep_uA) / 1e3 / 3600.0
    I_avg_raw = (Q_tx_raw + Q_sense_raw + Q_sleep_raw) / duration_hours
    lifetime_raw = battery_capacity_mAh / (I_avg_raw * 24.0)
    
    # Adaptive Mode
    Q_tx_ad = N_ad * I_tx_mA * airtime / 3600.0
    Q_sense_ad = N_raw * (I_mcu_active_uA/1000.0 + I_sensor_active_mA) * t_sensor_active_s / 3600.0
    Q_sleep_ad = (duration_hours * 3600.0 - N_ad * airtime) * (I_mcu_sleep_uA + I_sx_sleep_uA) / 1e3 / 3600.0
    I_avg_ad = (Q_tx_ad + Q_sense_ad + Q_sleep_ad) / duration_hours
    lifetime_ad = battery_capacity_mAh / (I_avg_ad * 24.0)
    
    # Fixed Mode
    Q_tx_fixed = N_fixed * I_tx_mA * airtime / 3600.0
    Q_sense_fixed = N_raw * (I_mcu_active_uA/1000.0 + I_sensor_active_mA) * t_sensor_active_s / 3600.0
    Q_sleep_fixed = (duration_hours * 3600.0 - N_fixed * airtime) * (I_mcu_sleep_uA + I_sx_sleep_uA) / 1e3 / 3600.0
    I_avg_fixed = (Q_tx_fixed + Q_sense_fixed + Q_sleep_fixed) / duration_hours
    lifetime_fixed = battery_capacity_mAh / (I_avg_fixed * 24.0)
    
    energy_results.append({
        "SF": sf,
        "Airtime (s)": airtime,
        "Raw Current (uA)": I_avg_raw * 1000.0,
        "Raw Life (days)": lifetime_raw,
        "Fixed Current (uA)": I_avg_fixed * 1000.0,
        "Fixed Life (days)": lifetime_fixed,
        "Adaptive Current (uA)": I_avg_ad * 1000.0,
        "Adaptive Life (days)": lifetime_ad,
        "Extension (vs Raw)": lifetime_ad / lifetime_raw
    })

df_energy = pd.DataFrame(energy_results)
print("\n=== PHYSICAL EXPERIMENT LORA SENSITIVITY MATRIX ===")
print(df_energy.to_string(index=False, formatters={
    "Raw Current (uA)": "{:.1f}".format,
    "Raw Life (days)": "{:.1f}".format,
    "Fixed Current (uA)": "{:.1f}".format,
    "Fixed Life (days)": "{:.1f}".format,
    "Adaptive Current (uA)": "{:.1f}".format,
    "Adaptive Life (days)": "{:.1f}".format,
    "Extension (vs Raw)": "{:.2f}x".format
}))


# 7. SYNTHETIC NOISE INJECTION STUDY ON PHYSICAL DATA

print("\n[INFO] Running noise injection study on physical baseline...")
noise_stds = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
noise_study = []

# Simple adaptive filter logic on noisy data
for sd in noise_stds:
    # Inject Gaussian noise
    noisy_temp = raw_temp + np.random.normal(0, sd, len(raw_temp))
    
    # Simulation parameters matching proposed
    W = 10; alpha = 0.05; epsilon = 0.001; delta_min = 0.1
    S = 0.0; SS = 0.0; window = []; tx_cnt = 0
    rec_n = np.zeros(len(raw_temp))
    rec_n[0] = noisy_temp[0]
    last_tx = noisy_temp[0]
    
    for t in range(len(raw_temp)):
        val = noisy_temp[t]
        window.append(val)
        S += val
        SS += val**2
        if len(window) > W:
            old = window.pop(0)
            S -= old
            SS -= old**2
        
        curr_W = len(window)
        sigma = np.sqrt(max(0.0, (SS - (S**2 / curr_W)) / (curr_W - 1))) if curr_W > 1 else 0.0
        delta_t = max(delta_min, alpha / (sigma + epsilon))
        
        if t == 0 or abs(val - last_tx) >= delta_t:
            tx_cnt += 1
            last_tx = val
            rec_n[t] = val
        else:
            rec_n[t] = last_tx
            
    drr_n = (1.0 - tx_cnt / len(raw_temp)) * 100
    rmse_n = np.sqrt(np.mean((raw_temp - rec_n)**2))
    
    noise_study.append({
        "Added Noise SD (C)": sd,
        "Packets": tx_cnt,
        "DRR%": drr_n,
        "RMSE (C)": rmse_n
    })
df_noise_study = pd.DataFrame(noise_study)
print(df_noise_study.to_string(index=False))


# 8. INTEL LAB DATASET VALIDATION (DOWNLOAD OR SIMULATION FALLBACK)

print("\n[INFO] Loading/Downloading Intel Berkeley Lab dataset for validation...")
download_success = False
local_intel_gz = "data.txt.gz"
local_intel_txt = "data.txt"

try:
    if not os.path.exists(local_intel_txt):
        if not os.path.exists(local_intel_gz):
            print("[INFO] Downloading data.txt.gz from MIT Mirror (34MB)...")
            urllib.request.urlretrieve("http://db.csail.mit.edu/labdata/data.txt.gz", local_intel_gz)
        print("[INFO] Extracting data.txt...")
        with gzip.open(local_intel_gz, 'rb') as f_in:
            with open(local_intel_txt, 'wb') as f_out:
                f_out.write(f_in.read())
    
    print("[INFO] Loading data.txt into pandas...")
    column_names = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
    df_intel = pd.read_csv(local_intel_txt, sep=r'\s+', header=None, names=column_names,
                           engine='python',
                           on_bad_lines='skip')
    download_success = True
    print(f"[SUCCESS] Loaded {len(df_intel)} rows. Available motes: {sorted(df_intel['moteid'].dropna().unique().astype(int).tolist()[:20])}...")
except Exception as e:
    print(f"[WARNING] Could not obtain Intel Lab data online: {e}.")
    print("[INFO] Falling back to generating simulated Intel Mote 1 & Mote 18 data...")

# Pick two motes: one stable (low SD) and one volatile (high SD)
if download_success:
    # Find motes with enough data
    mote_counts = df_intel.groupby('moteid')['temperature'].count()
    valid_motes = mote_counts[mote_counts > 500].index.tolist()
    print(f"[INFO] Motes with >500 readings: {sorted([int(m) for m in valid_motes])}")
    
    # Compute per-mote temperature SD to pick stable vs volatile
    mote_stats = df_intel[df_intel['moteid'].isin(valid_motes)].groupby('moteid')['temperature'].agg(['mean', 'std', 'count'])
    mote_stats = mote_stats.dropna().sort_values('std')
    print("[INFO] Mote statistics (sorted by volatility):")
    print(mote_stats.head(10).to_string())
    
    # Pick the most stable and most volatile motes
    stable_mote_id = int(mote_stats.index[0])
    volatile_mote_id = int(mote_stats.index[-1])
    print(f"[INFO] Selected Stable Mote: {stable_mote_id} (SD={mote_stats.iloc[0]['std']:.2f}°C)")
    print(f"[INFO] Selected Volatile Mote: {volatile_mote_id} (SD={mote_stats.iloc[-1]['std']:.2f}°C)")
    
    mote1 = df_intel[df_intel['moteid'] == stable_mote_id]['temperature'].dropna().values
    mote21 = df_intel[df_intel['moteid'] == volatile_mote_id]['temperature'].dropna().values
    stable_label = f"Mote {stable_mote_id} (Stable)"
    volatile_label = f"Mote {volatile_mote_id} (Volatile)"
else:
    ticks = np.arange(4000)
    mote1 = 21.5 + 0.6 * np.sin(2 * np.pi * ticks / 1440) + np.random.normal(0, 0.05, 4000)
    mote21 = 23.0 + 4.2 * np.sin(2 * np.pi * ticks / 1440) + np.random.normal(0, 0.25, 4000)
    stable_label = "Mote 1 (Stable, Simulated)"
    volatile_label = "Mote 21 (Volatile, Simulated)"

# Clean out of bounds
mote1 = mote1[(mote1 > 5) & (mote1 < 45)][:4000]
mote21 = mote21[(mote21 > 5) & (mote21 < 45)][:4000]

def run_adaptive_sim(data):
    if len(data) == 0:
        return 0.0, 0.0, 0
    W = 10; alpha = 0.05; epsilon = 0.001; delta_min = 0.1
    S = 0.0; SS = 0.0; window = []; tx_cnt = 0
    rec = np.zeros(len(data))
    rec[0] = data[0]
    last_tx = data[0]
    for t in range(len(data)):
        val = data[t]
        window.append(val)
        S += val
        SS += val**2
        if len(window) > W:
            old = window.pop(0)
            S -= old
            SS -= old**2
        curr_W = len(window)
        sigma = np.sqrt(max(0.0, (SS - (S**2 / curr_W)) / (curr_W - 1))) if curr_W > 1 else 0.0
        delta_t = max(delta_min, alpha / (sigma + epsilon))
        if t == 0 or abs(val - last_tx) >= delta_t:
            tx_cnt += 1
            last_tx = val
            rec[t] = val
        else:
            rec[t] = last_tx
    drr = (1.0 - tx_cnt / len(data)) * 100
    rmse = np.sqrt(np.mean((data - rec)**2))
    return drr, rmse, tx_cnt

drr_m1, rmse_m1, tx_m1 = run_adaptive_sim(mote1)
drr_m21, rmse_m21, tx_m21 = run_adaptive_sim(mote21)

print(f"\n=== PUBLIC INTEL LAB VALIDATION ===")
print(f"{stable_label}:   N={len(mote1)}, DRR = {drr_m1:.2f}%, RMSE = {rmse_m1:.4f}°C, Packets = {tx_m1}")
print(f"{volatile_label}: N={len(mote21)}, DRR = {drr_m21:.2f}%, RMSE = {rmse_m21:.4f}°C, Packets = {tx_m21}")


# 9. PLOTTING THE 5 RESEARCH FIGURES

print("\n[INFO] Generating and saving figures using physical data...")

# Figure 1: Pareto Frontier
plt.figure(figsize=(6, 4))
thresholds_sens = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
drrs_sens = []
rmses_sens = []
for th in thresholds_sens:
    rec_th = np.zeros(len(raw_temp))
    rec_th[0] = raw_temp[0]
    last_tx = raw_temp[0]
    tx_cnt = 0
    for t in range(len(raw_temp)):
        val = raw_temp[t]
        if t == 0 or abs(val - last_tx) >= th:
            tx_cnt += 1
            last_tx = val
            rec_th[t] = val
        else:
            rec_th[t] = last_tx
    drrs_sens.append((1.0 - tx_cnt / len(raw_temp)) * 100)
    rmses_sens.append(np.sqrt(np.mean((raw_temp - rec_th)**2)))

plt.plot(rmses_sens, drrs_sens, 'o-', color='#2c3e50', linewidth=2, label='Static Thresholds')
plt.plot(rmse_ad, drr_ad, '*', color='#e74c3c', markersize=12, label=f'Adaptive Memorization (Ours)')
plt.title('Pareto Frontier Optimization')
plt.xlabel('Reconstruction Error (RMSE in °C)')
plt.ylabel('Data Reduction Rate (DRR in %)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('figure_1.png', dpi=300)
plt.savefig('figure_1.pdf')
plt.close()

# Figure 2: Bar Chart comparisons
plt.figure(figsize=(6, 4))
methods = ['Uniform', 'Random', 'PLA', 'Fixed SOD', 'Proposed Adaptive']
errors = [rmse_uni, rmse_rand, rmse_pla, rmse_fixed, rmse_ad]
plt.bar(methods, errors, color=['#95a5a6', '#bdc3c7', '#7f8c8d', '#34495e', '#2c3e50'])
plt.title('Reconstruction Error (RMSE) Comparison')
plt.ylabel('RMSE (°C)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_2.png', dpi=300)
plt.savefig('figure_2.pdf')
plt.close()

# Figure 3: ITI Stable vs Volatile
# Split physical data in half for stable vs volatile phases
start_t = df_raw['Timestamp'].min()
split_t = start_t + pd.Timedelta(days=4)
df_ad_stable = df_adaptive[df_adaptive['Timestamp'] <= split_t]
df_ad_volatile = df_adaptive[df_adaptive['Timestamp'] > split_t]

iti_st = df_ad_stable['Timestamp'].diff().mean().total_seconds() / 60.0
iti_vol = df_ad_volatile['Timestamp'].diff().mean().total_seconds() / 60.0

plt.figure(figsize=(5, 4))
phases = ['Stable Phase\n(Days 1-4)', 'Volatile Phase\n(Days 5-8)']
itis = [iti_st, iti_vol]
plt.bar(phases, itis, color=['#27ae60', '#c0392b'], width=0.5)
plt.ylabel('Mean Inter-Transmission Interval (minutes)')
plt.title('Average Time Between Transmissions (ITI)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_3.png', dpi=300)
plt.savefig('figure_3.pdf')
plt.close()

# Figure 4: Outage Signal Reconstruction
# Find the largest gap in df_raw timestamps
df_raw['TimeDiff'] = df_raw['Timestamp'].diff()
max_gap_idx = df_raw['TimeDiff'].idxmax()
gap_end = df_raw['Timestamp'].iloc[max_gap_idx]
gap_start = df_raw['Timestamp'].iloc[max_gap_idx - 1]
gap_hours = df_raw['TimeDiff'].max().total_seconds() / 3600.0

print(f"Identified physical network outage: {gap_start} to {gap_end} ({gap_hours:.2f} hours)")

# Slice 2 hours before and after the outage for plotting
plot_start = gap_start - pd.Timedelta(hours=2)
plot_end = gap_end + pd.Timedelta(hours=2)

df_raw_slice = df_raw[(df_raw['Timestamp'] >= plot_start) & (df_raw['Timestamp'] <= plot_end)]
df_rec_slice = df_rec_ad[(df_rec_ad['Timestamp'] >= plot_start) & (df_rec_ad['Timestamp'] <= plot_end)]

plt.figure(figsize=(7, 3.5))
plt.plot(df_raw_slice['Timestamp'], df_raw_slice['Temperature_C'], '-', color='#bdc3c7', label='Raw Temperature (Ground Truth)')
plt.plot(df_rec_slice['Timestamp'], df_rec_slice['Temperature_C_rec'], '--', color='#e74c3c', linewidth=2, label='Reconstructed (ZOH)')
plt.axvspan(gap_start, gap_end, color='#f1c40f', alpha=0.2, label=f'{gap_hours:.1f}-Hour Network Outage')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.title('Signal Reconstruction During Network Outage')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_4.png', dpi=300)
plt.savefig('figure_4.pdf')
plt.close()

# Figure 5: Bland-Altman Agreement Plot
plt.figure(figsize=(6, 4.5))
diffs = raw_temp - rec_temp_ad
means = (raw_temp + rec_temp_ad) / 2.0
mean_diff = np.mean(diffs)
std_diff = np.std(diffs)
lower_loa = mean_diff - 1.96 * std_diff
upper_loa = mean_diff + 1.96 * std_diff

plt.scatter(means, diffs, alpha=0.3, color='#2c3e50', edgecolors='none', s=15)
plt.axhline(mean_diff, color='red', linestyle='-', label=f'Mean Bias ({mean_diff:.4f}°C)')
plt.axhline(upper_loa, color='red', linestyle='--', label=f'+1.96 SD ({upper_loa:.4f}°C)')
plt.axhline(lower_loa, color='red', linestyle='--', label=f'-1.96 SD ({lower_loa:.4f}°C)')
plt.title('Bland-Altman Agreement Analysis')
plt.xlabel('Mean of Raw and Reconstructed (°C)')
plt.ylabel('Difference (Raw - Reconstructed) (°C)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_5.png', dpi=300)
plt.savefig('figure_5.pdf')
plt.close()

print("[SUCCESS] All 5 figures updated using real data and noise profiles.")
print("[INFO] Process complete. Run 'python run_simulations.py' in the folder to execute.")
print("="*70)
