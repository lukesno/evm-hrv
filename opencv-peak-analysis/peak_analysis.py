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

# Load the data from the specified .npz file
loaded_data = np.load(args.input_file)
red_intensity_over_time = loaded_data['red_intensity']
time = loaded_data['time']

# Perform peak detection
peaks, _ = find_peaks(red_intensity_over_time, height=args.height, distance=args.distance)

# Calculate time intervals between peaks
peak_time_intervals = np.diff(time[peaks])
# Print peak_time_intervals for verification
print("Peak Time Intervals:", peak_time_intervals)

# Extract HRV features using peak_time_intervals
mean_rr_interval = np.mean(peak_time_intervals)
std_rr_interval = np.std(peak_time_intervals)

# Sliding window for continuous HRV over time
window_size = 2  # Adjust the window size as needed
hrv_over_time = [np.std(peak_time_intervals[i:i+window_size]) for i in range(len(peak_time_intervals)-window_size+1)]
time_hrv = time[peaks][window_size-1:-1]  # Adjusted to ensure correct time points

# Convert HRV values to milliseconds
hrv_over_time_ms = np.array(hrv_over_time) * 1000
time_hrv_ms = np.array(time_hrv) * 1000

# Print HRV over time for verification
print("HRV Over Time:", hrv_over_time)
# Find minimum and maximum HRV values
min_hrv = np.min(hrv_over_time)
max_hrv = np.max(hrv_over_time)

# Print the minimum and maximum HRV values
print("Minimum HRV:", min_hrv)
print("Maximum HRV:", max_hrv)

# Plotting all figures in subplots
fig, axes = plt.subplots(1, 1, figsize=(10, 12))

# Plot HRV over time
axes.plot(time_hrv, hrv_over_time, label='HRV Over Time', color='purple')
# for discrete points
# axes.plot(time_hrv, hrv_over_time, 'o', label='HRV Over Time', color='purple')
axes.set_xlabel('Time (seconds)')
axes.set_ylabel('HRV (ms)')
axes.set_title(f'HRV Over Time (Window Size = {window_size})')
axes.legend()

# Save the plot as a .png file
output_png_path = 'hrv_over_time_plot.png'
plt.savefig(output_png_path)

plt.tight_layout()
plt.show()