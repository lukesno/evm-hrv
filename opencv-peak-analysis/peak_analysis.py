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

# Plot the detected peaks
plt.plot(time, red_intensity_over_time, label='Red Intensity Over Time', color='red')
plt.plot(time[peaks], red_intensity_over_time[peaks], 'x', label='Detected Peaks', color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Red Intensity')
plt.title('Detected Peaks in Red Intensity Over Time')
plt.legend()
plt.show()

# Calculate time intervals between peaks
peak_time_intervals = np.diff(time[peaks])

# Extract HRV features using peak_time_intervals

# Time-domain HRV features
mean_rr_interval = np.mean(peak_time_intervals)
std_rr_interval = np.std(peak_time_intervals)

# Frequency-domain HRV features (you may need additional libraries for spectral analysis)
# You can use methods like FFT to analyze the frequency content of the signal
# Frequency-domain HRV features using FFT
fft_result = fft(red_intensity_over_time) # fft is used to compute the FFT of the red intensity signal over time 
frequencies = np.fft.fftfreq(len(fft_result), d=1/fps)
# calculates the frequencies associated with the FFT result. It uses np.fft.fftfreq to
# generate an array of frequencies based on the length of the FFT result (len(fft_result)) 
# and the sampling rate (fps).
positive_frequencies = frequencies[:len(frequencies)//2]
fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
#  extracts the magnitude spectrum of the positive frequencies

# Plot the FFT result
# It visualizes how the magnitude of the signal varies across different frequencies.
plt.figure()
plt.plot(positive_frequencies, fft_magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Red Intensity Signal')
plt.show()

# Extract frequency-domain HRV features
# You can define specific frequency bands (e.g., LF, HF) and calculate power in each band
# Example: LF (low frequency) range [0.04, 0.15] Hz, HF (high frequency) range [0.15, 0.4] Hz
lf_indices = np.where((positive_frequencies >= 0.04) & (positive_frequencies < 0.15))[0]
hf_indices = np.where((positive_frequencies >= 0.15) & (positive_frequencies < 0.4))[0]

lf_power = np.sum(fft_magnitude[lf_indices])
hf_power = np.sum(fft_magnitude[hf_indices])


# Display the calculated time intervals between peaks and HRV features
print("Time Intervals between Peaks (seconds):", peak_time_intervals)
print("Mean RR Interval:", mean_rr_interval)
print("Standard Deviation of RR Intervals:", std_rr_interval)

# Visualize HRV features
plt.figure()
plt.bar(['Mean RR Interval', 'Standard Deviation of RR Intervals'], [mean_rr_interval, std_rr_interval], color=['blue', 'orange'])
plt.ylabel('Time (sec)')
plt.title('HRV Features')
plt.show()
