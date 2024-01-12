import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_red_channel_over_time(input_file, output_dir='', verbose=False):
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

    # Set output directory path
    if output_dir == '':
        output_dir = os.getcwd()

    filename_prefix = os.path.splitext(args.input_file)[0]
    filename_prefix = os.path.split(filename_prefix)[1]

    # Save time and red intensity data as .npz file in the working directory
    output_npy_path = os.path.join(output_dir, filename_prefix + '_light_intensity.npz')
    np.savez(output_npy_path, time=time, red_intensity=red_intensity_over_time)

    # Load the data from the .npz file
    loaded_data = np.load(output_npy_path)
    loaded_time = loaded_data['time']
    loaded_red_intensity = loaded_data['red_intensity']

    # Print the time and red intensity data points
    if verbose:
        print("Time Data Points:")
        print(loaded_time)
        print("Light Intensity Data Points:")
        print(loaded_red_intensity)
    
    # Plot the red intensity over time
    plt.plot(time, red_intensity_over_time, label='Light Intensity Over Time', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Light Intensity')
    video_filename = os.path.basename(args.input_file)
    plt.title(f'Light Intensity Over Time - {video_filename}')
    plt.legend()

    output_image_path = os.path.join(output_dir, filename_prefix + '_red_intensity_plot.png')
    
    plt.savefig(output_image_path)
    # Display the plot
    # plt.show()

    print(f"\nCurrent working directory '{os.getcwd()}'")
    print(f"Plot saved as '{output_image_path}'")
    print(f"Light intensity data saved as '{output_npy_path}'")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot Light intensity over time from a Light channel video.')
    parser.add_argument('input_file', type=str, help='Path to the Light channel video file')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to the output dir', default='')
    args = parser.parse_args()

    plot_red_channel_over_time(args.input_file, args.output_dir)