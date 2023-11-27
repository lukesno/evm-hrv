import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from scipy.fft import fft

# Parse command line arguments
parser = argparse.ArgumentParser(description='Perform peak detection and analyze red intensity over time.')
parser.add_argument('input_file', type=str, help='Path to the .npy file containing red intensity over time data')
parser.add_argument('--height', type=float, default=0.5, help='Peak height threshold for detection')
parser.add_argument('--distance', type=int, default=10, help='Minimum distance between peaks')
args = parser.parse_args()

# Load the red intensity data from the specified .npy file
red_intensity_over_time = np.load(args.input_file)

# Create time array
fps = 30  # Assuming 30 frames per second, adjust accordingly
num_frames = len(red_intensity_over_time)
time = np.arange(0, num_frames) / fps

# Perform peak detection
peaks, _ = find_peaks(red_intensity_over_time, height=args.height, distance=args.distance)

# Calculate time intervals between peaks
peak_time_intervals = np.diff(time[peaks])

# Extract HRV features using peak_time_intervals
mean_rr_interval = np.mean(peak_time_intervals)
std_rr_interval = np.std(peak_time_intervals)

# Frequency-domain HRV features using FFT
fft_result = fft(red_intensity_over_time)
frequencies = np.fft.fftfreq(len(fft_result), d=1/fps)
positive_frequencies = frequencies[:len(frequencies)//2]
fft_magnitude = np.abs(fft_result[:len(fft_result)//2])

# Extract frequency-domain HRV features
lf_indices = np.where((positive_frequencies >= 0.04) & (positive_frequencies < 0.15))[0]
hf_indices = np.where((positive_frequencies >= 0.15) & (positive_frequencies < 0.4))[0]
lf_power = np.sum(fft_magnitude[lf_indices])
hf_power = np.sum(fft_magnitude[hf_indices])

# Plotting all figures in subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot red intensity over time and detected peaks
axes[0, 0].plot(time, red_intensity_over_time, label='Red Intensity Over Time', color='red')
axes[0, 0].plot(time[peaks], red_intensity_over_time[peaks], 'x', label='Detected Peaks', color='blue')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Red Intensity')
axes[0, 0].set_title('Detected Peaks in Red Intensity Over Time')
axes[0, 0].legend()

# Plot Frequency Spectrum of Red Intensity Signal
axes[0, 1].plot(positive_frequencies, fft_magnitude)
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].set_title('Frequency Spectrum of Red Intensity Signal')

# Plot Histogram of Time Intervals between Peaks
axes[1, 0].hist(peak_time_intervals, bins=30, color='green', alpha=0.7)
axes[1, 0].set_xlabel('Time Intervals between Peaks (seconds)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram of Time Intervals between Peaks')

# Plot Bar Chart of HRV Features
axes[1, 1].bar(['Mean RR Interval', 'Standard Deviation of RR Intervals'], [mean_rr_interval, std_rr_interval], color=['blue', 'orange'])
axes[1, 1].set_ylabel('Time (sec)')
axes[1, 1].set_title('HRV Features')

plt.tight_layout()
plt.show()
