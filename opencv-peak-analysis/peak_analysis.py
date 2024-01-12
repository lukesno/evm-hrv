import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from scipy.fft import fft
import os


def peak_analysis(input_file, output_dir, height, distance):
    # Load the data from the specified .npz file
    loaded_data = np.load(input_file)
    red_intensity_over_time = loaded_data['red_intensity']
    time = loaded_data['time']

    # Perform peak detection
    peaks, _ = find_peaks(red_intensity_over_time, height=height, distance=distance)

    # Calculate time intervals between peaks
    peak_time_intervals = np.diff(time[peaks])
    # Print peak_time_intervals for verification
    print("Peak Time Intervals:", peak_time_intervals)

    # Extract HRV features using peak_time_intervals
    mean_rr_interval = np.mean(peak_time_intervals)
    std_rr_interval = np.std(peak_time_intervals)
    peak_time_intervals_ms = np.array(peak_time_intervals) * 1000

    # Sliding window for continuous HRV over time
    window_size = 10 # Adjust the window size as needed
    SDNN  = [np.std(peak_time_intervals_ms[i:i+window_size]) for i in range(len(peak_time_intervals_ms)-window_size+1)]
    SD    = [(peak_time_intervals_ms[i] - peak_time_intervals_ms[i+1]) for i in range(len(peak_time_intervals_ms)-1)]
    SSD   = np.square(SD)
    MSSD  = np.mean(SSD) 
    RMSSD = math.sqrt(MSSD)

    hrv_over_time = SDNN
    
    time_hrv = time[peaks][window_size-1:-1]  # Adjusted to ensure correct time points

    # Print HRV over time for verification
    print("SDNN over time:", hrv_over_time)
    # Find minimum and maximum HRV values
    min_hrv = np.min(hrv_over_time)
    max_hrv = np.max(hrv_over_time)

    # Print the minimum and maximum HRV values
    print("Minimum HRV:", min_hrv)
    print("Maximum HRV:", max_hrv)

    # Plotting all figures in subplots
    fig, axes = plt.subplots(1, 1, figsize=(10, 12))

    # Plot HRV over time
    axes.plot(time_hrv, hrv_over_time, label='SDNN', color='purple')
    # for discrete points
    # axes.plot(time_hrv, hrv_over_time, 'o', label='HRV Over Time', color='purple')
    axes.set_xlabel('Time (seconds)')
    axes.set_ylabel('SDNN (ms)')
    axes.set_title(f'Standard Deviation of RR Intervals (STNN) (Window Size = {window_size}) RMSSD = {RMSSD}')
    axes.legend()

    # Set output directory path
    if output_dir == '':
        output_dir = os.getcwd()

    filename_prefix = os.path.splitext(input_file)[0]
    filename_prefix = os.path.split(filename_prefix)[1]
    # Save time and red intensity data as .npz file in the working directory
    output_png_path= os.path.join(output_dir, filename_prefix + '_hrv_over_time_plot.png')
    # Save the plot as a .png file
    plt.savefig(output_png_path)
    print(f"Plot saved as '{output_png_path}'")

    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform peak detection and analyze red intensity over time.')
    parser.add_argument('input_file', type=str, help='Path to the .npy file containing red intensity over time data')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to the output dir', default='')
    parser.add_argument('--height', type=float, default=0.5, help='Peak height threshold for detection')
    parser.add_argument('--distance', type=int, default=10, help='Minimum distance between peaks')
    args = parser.parse_args()

    peak_analysis(args.input_file, args.output_dir, args.height, args.distance)