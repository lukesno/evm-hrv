import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot red intensity over time from a red channel video.')
parser.add_argument('input_file', type=str, help='Path to the red channel video file')
args = parser.parse_args()

# Load the video file
video_capture = cv2.VideoCapture(args.input_file)

# Get video details
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an array to store red intensity over time
red_intensity_over_time = []

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Extract the red channel from the frame
    red_channel = frame[:, :, 2]  # Assuming the red channel is the third channel (0-indexed)

    # Calculate the average red intensity of the frame
    red_intensity_frame = np.mean(red_channel)

    # Update the red intensity array
    red_intensity_over_time.append(red_intensity_frame)

# Release video capture object
video_capture.release()

# Create time array
time = np.arange(0, num_frames) / fps

# Save time and red intensity data as .npz file in the working directory
output_npy_path = os.path.join(os.getcwd(), os.path.splitext(args.input_file)[0] + '_red_intensity.npz')
np.savez(output_npy_path, time=time, red_intensity=red_intensity_over_time)

# Load the data from the .npz file
loaded_data = np.load(output_npy_path)
loaded_time = loaded_data['time']
loaded_red_intensity = loaded_data['red_intensity']

# Print the time and red intensity data points
print("Time Data Points:")
print(loaded_time)
print("Red Intensity Data Points:")
print(loaded_red_intensity)

# Plot the red intensity over time
plt.plot(time, red_intensity_over_time, label='Red Intensity Over Time', color='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Red Intensity')
video_filename = os.path.basename(args.input_file)
plt.title(f'Red Intensity Over Time - {video_filename}')
plt.legend()

# Save the plot as an image in the working directory
output_image_path = os.path.join(os.getcwd(), os.path.splitext(args.input_file)[0] + '_red_intensity_plot.png')
plt.savefig(output_image_path)

# Display the plot
plt.show()

print(f"Plot saved as '{output_image_path}'")
print(f"Red intensity data saved as '{output_npy_path}'")
