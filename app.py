import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



# --------------------------------
# STEP 1: DATA PREPARATION
# --------------------------------
fs = 100  # Sampling frequency (Hz)
t = np.arange(0, 20, 1/fs)

# Healthy bridge vibration
f1, f2 = 5, 12
x_healthy = 0.6*np.sin(2*np.pi*f1*t) + 0.4*np.sin(2*np.pi*f2*t)
x_healthy += 0.05*np.random.randn(len(t))

# Damaged bridge vibration
f1_d, f2_d = 4.5, 11
x_damaged = 0.6*np.sin(2*np.pi*f1_d*t) + 0.4*np.sin(2*np.pi*f2_d*t)
x_damaged += 0.05*np.random.randn(len(t))

# --------------------------------
# STEP 2: DETREND
# --------------------------------
x_h = signal.detrend(x_healthy)
x_d = signal.detrend(x_damaged)

# -------------------------------
# STEP 3: BAND-PASS FILTER (FIXED)
# --------------------------------
low = 1
high = 40  # MUST be < fs/2
nyq = fs / 2

b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
x_h_filt = signal.filtfilt(b, a, x_h)
x_d_filt = signal.filtfilt(b, a, x_d)

# --------------------------------
# STEP 4: PSD USING WELCH METHOD
# --------------------------------
f_h, Pxx_h = signal.welch(x_h_filt, fs, nperseg=1024)
f_d, Pxx_d = signal.welch(x_d_filt, fs, nperseg=1024)

# --------------------------------
# STEP 5: MODAL ANALYSIS (PEAK DETECTION)
# --------------------------------
peaks_h, _ = signal.find_peaks(Pxx_h, height=0.0001)
peaks_d, _ = signal.find_peaks(Pxx_d, height=0.0001)

f_base = f_h[peaks_h][:2]
f_curr = f_d[peaks_d][:2]

# --------------------------------
# STEP 6: DAMAGE DETECTION
# --------------------------------
threshold = 0.05
damage_ratio = np.abs(f_curr - f_base) / f_base

if np.any(damage_ratio > threshold):
    status = "⚠️ DAMAGE DETECTED"
else:
    status = "✅ HEALTHY"

# --------------------------------
# STEP 7: HEALTH INDEX
# --------------------------------
health_index = 1 - np.mean(damage_ratio)

# --------------------------------
# STEP 8: VISUALIZATION
# --------------------------------
plt.figure()
plt.semilogy(f_h, Pxx_h, label="Healthy PSD")
plt.semilogy(f_d, Pxx_d, label="Damaged PSD")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("Power Spectral Density Comparison")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.bar(["Mode 1", "Mode 2"], f_base, label="Healthy")
plt.bar(["Mode 1", "Mode 2"], f_curr, alpha=0.7, label="Damaged")
plt.ylabel("Frequency (Hz)")
plt.title("Natural Frequency Shift")
plt.legend()
plt.show()

# --------------------------------
# STEP 9: OUTPUT
# --------------------------------
print("\n==== RESULTS ====")
print("Baseline Frequencies (Hz):", f_base)
print("Current Frequencies (Hz):", f_curr)
print("Damage Ratio:", damage_ratio)
print("Health Index:", round(health_index, 3))
print("Status:", status)
