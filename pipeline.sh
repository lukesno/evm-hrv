#!/bin/bash

# Assuming the GitHub repository is cloned to /opencv-head-detector
parent_dir="/opencv-head-detector "

# Change to the head-detector-opencv directory
cd "/head-detector-opencv"

# Run the head-detector-opencv script
python auto_lock_detection.py

# Capture the output video filename
output_video="luke_face_after_EVM_after_red_channel_extraction.mp4"

# Change to the EVM directory
cd "../EVM/EVM_Matlab_bin-1.1-linux64"

# Run the EVM script with the output video from the previous step
./run_evm.sh EVM_BIN face_output.mp4 res 30 'color' 140/60 160/60 150 'ideal' 1 6
evm_output="face_output.mp4"

# Change to the opencv-red-channel-extraction directory
cd "../opencv-red-channel-extraction"

# Run the red channel extraction script with the output from the EVM script
python extract_red_channel.py "${evm_output}"

# Capture the output filename
red_channel_output="luke_face_after_EVM_after_red_channel_extraction_red_intensity.npy"

# Change to the opencv-red-time-plot directory
cd "../opencv-red-time-plot"

# Run the time plot script with the red channel output
python plot_red_channel_over_time.py "../head-detector-opencv/${output_video}"

# Capture the output filename
time_plot_output="luke_face_after_EVM_after_red_channel_extraction_red_intensity.npy"

# Change to the opencv-peak-analysis directory
cd "../opencv-peak-analysis"

# Run the peak analysis script with the time plot output
python peak_analysis.py "../head-detector-opencv/${time_plot_output}" --height 0.8 --distance 20
