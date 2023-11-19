import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot red intensity over time from a red channel video.')
parser.add_argument('input_file', type=str, help='Path to the red channel video file')
args = parser.parse_args()

# Load the video file with only the red channel
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

    # Calculate the average red intensity of the frame
    red_intensity_frame = np.mean(frame)

    # Update the red intensity array
    red_intensity_over_time.append(red_intensity_frame)

# Release video capture object
video_capture.release()

# Create time array
time = np.arange(0, num_frames) / fps

# Save red intensity data as .npy file
output_npy_path = os.path.splitext(args.input_file)[0] + '_red_intensity.npy'
np.save(output_npy_path, red_intensity_over_time)

# Plot the red intensity over time
plt.plot(time, red_intensity_over_time, label='Red Intensity Over Time', color='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Red Intensity')
plt.title('Red Intensity Over Time')
plt.legend()

# Save the plot as an image
output_image_path = os.path.splitext(args.input_file)[0] + '_red_intensity_plot.png'
plt.savefig(output_image_path)

# Display the plot
plt.show()

print(f"Plot saved as '{output_image_path}'")
print(f"Red intensity data saved as '{output_npy_path}'")
