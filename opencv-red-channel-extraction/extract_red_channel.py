import cv2
import argparse
import os
import numpy as np

def smooth_data(data, window_size):
    # Apply a simple moving average to smooth the data along each row
    smoothed_data = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        smoothed_data[i, :] = np.convolve(data[i, :], np.ones(window_size)/window_size, mode='same')

    return smoothed_data.astype(np.uint8)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract the red channel from a video file.')
parser.add_argument('input_file', type=str, help='Path to the input video file')
args = parser.parse_args()

# Check if the input file has an AVI extension
if args.input_file.lower().endswith('.avi'):
    # Load the AVI video file
    video_capture = cv2.VideoCapture(args.input_file)

    # Get video details
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_file = 'output_red_channel.mp4'
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Extract the red channel
        red_channel = frame[:, :, 2]  # Assuming the order is BGR, with red at index 2

        # Smooth the red channel data
        smoothed_red_channel = smooth_data(red_channel, window_size=5)  # Adjust window_size as needed

        # Merge the smoothed red channel into a 3-channel image
        smoothed_red_frame = cv2.merge([smoothed_red_channel, smoothed_red_channel, smoothed_red_channel])

        # Write the frame to the output video
        video_writer.write(smoothed_red_frame)

    # Release video capture and writer objects
    video_capture.release()
    video_writer.release()

    print(f"Red channel extraction with data smoothing completed. Output saved as '{output_file}'")
