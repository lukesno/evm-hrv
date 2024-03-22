#===========================================================
# Import necessary libraries
import cv2
import dlib
from PIL import Image
import numpy as np
from collections import deque
from scipy.signal import find_peaks, butter, sosfilt
import matplotlib.pyplot as plt
import os
import math
import time
#===========================================================

#===========================================================
# Constants
DO_WRITE_VIDEO = False
FACIAL_PIXEL_DEVIATION_THRESHOLD = 5
FRAMES_PER_FACIAL_DETECTION = 5 
#===========================================================

#===========================================================
# Set up
# Load the pre-trained face detector from dlib
# Load the facial landmark predictor from dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Set the desired size for the windows
cv2.namedWindow("Facial Landmarks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Facial Landmarks", 800, 600)  # Adjust the size as needed
cheek_frame_size = 300
cv2.namedWindow("Left Cheek", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Cheek", cheek_frame_size, cheek_frame_size)  # Adjust the size as needed
cv2.namedWindow("Right Cheek", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Cheek", cheek_frame_size, cheek_frame_size)  # Adjust the size as needed

# Open the camera
src = "vids/aaron_nap_20240313.mp4"
cap = cv2.VideoCapture(0)
fps    = 10
# Check if camera or videofile (src is integer or string)
if isinstance(src, int):
    is_camera = True
else:
    is_camera = False
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a new direct named 'out' under vids, if 'out' does not exist
if not os.path.exists("vids/out"):
    os.makedirs("vids/out")

# Create VideoWriter object to save the output
if DO_WRITE_VIDEO:
    fourcc          = cv2.VideoWriter_fourcc(*'mp4v')
    face_vid_writer = cv2.VideoWriter("vids/out/whole_face_output.mp4", fourcc, fps, (width, height), isColor=True)
    lc_vid_writer   = cv2.VideoWriter("vids/out/left_cheek_output.mp4", fourcc, fps, (cheek_frame_size, cheek_frame_size), isColor=True)
    rc_vid_writer   = cv2.VideoWriter("vids/out/right_cheek_output.mp4", fourcc, fps, (cheek_frame_size, cheek_frame_size), isColor=True)
#===========================================================

#===========================================================
# Init Variables 
# Create an array to store red intensity over time
red_intensity_over_time     = []
lc_red_intensity_over_time  = []
rc_red_intensity_over_time  = []
lrc_red_intensity_over_time = []

print(f"Video Details: FPS: {fps}, Width: {width}, Height: {height}")
# Initialize with white pixels (for RGB images)
blank_frame   = np.ones((height, width, 3), dtype=np.uint8) * 255  
blank_image   = Image.fromarray(blank_frame, 'RGB')
blank_image   = cv2.resize(blank_frame, (cheek_frame_size, cheek_frame_size))
prev_rc_roi   = blank_frame
prev_lc_roi   = blank_frame
prev_lc_image = blank_image
prev_rc_image = blank_image

landmarks    = None
LC_top_cur   = 0
LC_bot_cur   = 0
LC_left_cur  = 0 
LC_right_cur = 0

RC_top_cur   = 0
RC_bot_cur   = 0
RC_left_cur  = 0
RC_right_cur = 0 

# Create zeros for initial light intensity
lc_red_intensity    = 0
rc_red_intensity    = 0
lrc_red_intensity   = 0
red_intensity_frame = 0
num_frames = 0
winSize             = 100
runningTime         = deque([0] * winSize, maxlen=winSize)
lc_last30seconds    = deque([0] * winSize, maxlen=winSize)
rc_last30seconds    = deque([0] * winSize, maxlen=winSize)
frame_last30seconds = deque([0] * winSize, maxlen=winSize)
lrc_last30seconds   = deque([0] * winSize, maxlen=winSize)

# Set up deques for rolling bpm averages
lc_bpm    = deque([0] * winSize, maxlen=winSize)
rc_bpm    = deque([0] * winSize, maxlen=winSize)
frame_bpm = deque([0] * winSize, maxlen=winSize)
lrc_bpm   = deque([0] * winSize, maxlen=winSize)
#================================================================

#===========================================================
# Function Definitions
def fft_on_deque(input_deque, runningTime):
    # Convert deque to numpy array
    input_array = np.array(input_deque)

    sample_rate  = len(input_array) / runningTime
    sample_space = 1 / sample_rate

    # Normalize the input array
    count = 0
    rollingAve = 0
    for i, data in enumerate(input_array):
        count      += 1
        rollingAve = rollingAve * (count - 1) / count + data / count
        input_array[i] -= rollingAve

    # Bandpass filter the input array
    sos         = butter(3, [1, 2.33], "bandpass", output='sos', fs=sample_rate)
    filteredSig = sosfilt(sos, input_array)

    # Perform FFT
    yf = np.fft.fft(input_array)
    xf = np.fft.fftfreq(len(input_array), sample_space) # Assuming the signal was sampled for 30 seconds
    
    # Get indices of frequency between 1 and 2.33 Hz
    pos_freq_indices = np.where((xf > 1) & (xf < 2.33))
    # get index of max yf given positive frequencies
    max_yf_index = np.argmax(np.abs(yf[pos_freq_indices]))
    max_freq = xf[pos_freq_indices][max_yf_index]

    return max_freq

def peak_analysis(input_file, output_dir, height, distance):
    
    filename_prefix = os.path.splitext(input_file)[0]
    filename_prefix = os.path.split(filename_prefix)[1]

    # Load the data from the specified .npz file
    loaded_data = np.load(input_file)
    intensity_over_time = loaded_data['intensity']
    time = loaded_data['time']  
    # print(red_intensity_over_time)
    count     = 0
    rollingAve = 0

    for i, data in enumerate(intensity_over_time):
        count      += 1
        rollingAve = rollingAve * (count - 1) / count + data / count
        intensity_over_time[i] -= rollingAve
        intensity_over_time[i] *= 100

    sos         = butter(3, [1, 2.33], "bandpass", output='sos', fs=15)
    filteredSig = sosfilt(sos, intensity_over_time) 

    # Perform peak detection
    peaks, _ = find_peaks(filteredSig, height=height, distance=distance)

    # Plot the red intensity over time
    plt.plot(time, filteredSig, label='Red Intensity')

    # Mark the detected peaks on the plotw
    plt.plot(time[peaks], filteredSig[peaks], 'rx', label='Peaks')

    # Customize the plot
    plt.title(f'{filename_prefix} Intensity with Detected Peaks')
    plt.xlabel('Time')
    plt.ylabel('Red Intensity')
    plt.legend()

    # Save time and red intensity data as .npz file in the working directory
    output_peak_png_path= os.path.join(output_dir, filename_prefix + '_labeled_peaks_plot.png')
    # Save the plot as a .png file
    plt.savefig(output_peak_png_path)
    print(f"Peak Plots saved as '{output_peak_png_path}'")
    plt.clf()

    # Calculate time intervals between peaks
    peak_time_intervals = np.diff(time[peaks])
     # Print peak_time_intervals for verification
    # print("Peak Time Intervals:", peak_time_intervals)

    # # Extract HRV features using peak_time_intervals
    # mean_rr_interval = np.mean(peak_time_intervals)
    # std_rr_interval = np.std(peak_time_intervals)
    # peak_time_intervals_ms = np.array(peak_time_intervals) * 1000

    # # Sliding window for continuous HRV over time
    # window_size = 10# Adjust the window size as needed
    # SDNN  = [np.std(peak_time_intervals_ms[i:i+window_size]) for i in range(len(peak_time_intervals_ms)-window_size+1)]
    # SD    = [(peak_time_intervals_ms[i] - peak_time_intervals_ms[i+1]) for i in range(len(peak_time_intervals_ms)-1)]
    # SSD   = np.square(SD)
    # MSSD  = np.mean(SSD) 
    # RMSSD = math.sqrt(MSSD)

    # hrv_over_time = SDNN
    
    # time_hrv = time[peaks][window_size-1:-1]  # Adjusted to ensure correct time points

    # # Print HRV over time for verification
    # print("SDNN over time:", hrv_over_time)
    # # Find minimum and maximum HRV values
    # min_hrv = np.min(hrv_over_time)
    # max_hrv = np.max(hrv_over_time)

    # # Print the minimum and maximum HRV values
    # print("Minimum HRV:", min_hrv)
    # print("Maximum HRV:", max_hrv)

    print(f"Number Peaks: {len(peaks)}")
    # # Plotting all figures in subplots
    # fig, axes = plt.subplots(1, 1, figsize=(10, 12))

    # # Plot HRV over time
    # axes.plot(time_hrv, hrv_over_time, label='SDNN', color='purple')
    # # for discrete points
    # # axes.plot(time_hrv, hrv_over_time, 'o', label='HRV Over Time', color='purple')
    # axes.set_xlabel('Time (seconds)')
    # axes.set_ylabel('SDNN (ms)')
    # axes.set_title(f'Standard Deviation of RR Intervals (STNN) (Window Size = {window_size}) RMSSD = {RMSSD}')
    # axes.legend()

    # # Set output directory path
    # if output_dir == '':
    #     output_dir = os.getcwd()

    # # Save tim8and red intensity data as .npz file in the working directory
    # output_png_path= os.path.join(output_dir, filename_prefix + '_hrv_over_time_plot.png')
    # # Save the plot as a .png file
    # plt.savefig(output_png_path)
    # print(f"Plot saved as '{output_png_path}'")

    # plt.tight_layout()
    # plt.show()
#================================================================
# MAIN
#================================================================
# start measuing the time of the program
time_start                       = time.time()    
frames_since_last_face_detection = FRAMES_PER_FACIAL_DETECTION

while True:
    # Capture frame-by-frame
    ret, frame_clean    = cap.read()
    if is_camera:
        start_of_frame_time = time.time()
    else:
        start_of_frame_time = 1/fps * num_frames
    num_frames += 1
    runningTime.append(start_of_frame_time)

    frame = frame_clean.copy()

    if frames_since_last_face_detection == FRAMES_PER_FACIAL_DETECTION:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        red_intensity_frame  = np.mean(frame[:, :, 2])

        # Detect faces in the frame
        faces = face_detector(gray)
        for face_num, face in enumerate(faces):
            # Get the facial landmarks
            landmarks = landmark_predictor(gray, face)

            # Calculate pixels corresponding to left cheek
            LC_top   = landmarks.part(36).y
            LC_bot   = landmarks.part(31).y
            LC_left  = landmarks.part(36).x 
            LC_right = landmarks.part(31).x

            LC_top   += (LC_bot - LC_top)//3
            LC_bot   += (LC_bot - LC_top)//3
            LC_left  -= (LC_right - LC_left)//3
            LC_right -= (LC_right - LC_left)//3

            # Calculate pixels corresponding to right cheek
            RC_top   = landmarks.part(45).y
            RC_bot   = landmarks.part(35).y
            RC_left  = landmarks.part(35).x
            RC_right = landmarks.part(45).x 

            RC_top   += (RC_bot - RC_top)//3
            RC_bot   += (RC_bot - RC_top)//3
            RC_right += (RC_right - RC_left)//3
            RC_left  += (RC_right - RC_left)//3

            # Update the current cheek positions if deviation is large
            if (
                abs(LC_top - LC_top_cur) > FACIAL_PIXEL_DEVIATION_THRESHOLD
                or
                abs(LC_right - LC_right_cur) > FACIAL_PIXEL_DEVIATION_THRESHOLD
            ):
                LC_top_cur   = LC_top
                LC_bot_cur   = LC_bot
                LC_left_cur  = LC_left
                LC_right_cur = LC_right 
            if (
                abs(RC_top - RC_top_cur) > FACIAL_PIXEL_DEVIATION_THRESHOLD
                or
                abs(RC_right - RC_right_cur) > FACIAL_PIXEL_DEVIATION_THRESHOLD
            ):
                RC_top_cur   = RC_top
                RC_bot_cur   = RC_bot
                RC_left_cur  = RC_left
                RC_right_cur = RC_right

            eyebrow_top = min(landmarks.part(19).y, landmarks.part(24).y)

            # # Draw landmarks on the face
            # for i in range(68):
            #     x, y = landmarks.part(i).x, landmarks.part(i).y
            #     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            #     cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            # One face only
            break
        frames_since_last_face_detection = 0
    else:
        frames_since_last_face_detection += 1

    lc_roi = frame_clean[LC_top_cur:LC_bot_cur, LC_left_cur:LC_right_cur]
    rc_roi = frame_clean[RC_top_cur:RC_bot_cur, RC_left_cur:RC_right_cur]

    lc_red_intensity   = np.mean(lc_roi[:, :, 2])
    rc_red_intensity   = np.mean(rc_roi[:, :, 2])
    lrc_red_intensity  = (lc_red_intensity + rc_red_intensity) / 2

    # Draw rectangles around cheeks
    cv2.rectangle(frame, (LC_left_cur, LC_top_cur), (LC_right_cur, LC_bot_cur), (0, 0, 255), 2)
    cv2.rectangle(frame, (RC_left_cur, RC_top_cur), (RC_right_cur, RC_bot_cur), (0, 0, 255), 2)
    # Draw box censor around the eyes and eyebrows
    if landmarks is not None:
        cv2.rectangle(frame, (landmarks.part(1).x, eyebrow_top), (landmarks.part(15).x, landmarks.part(29).y), (0,0,0), -1)
    
    # Update images if they are not empty
    if lc_roi.size:
        lc_image = Image.fromarray(lc_roi, 'RGB')
        lc_image = cv2.resize(lc_roi, (cheek_frame_size, cheek_frame_size))
        prev_lc_image = lc_image
    if rc_roi.size:
        rc_image = Image.fromarray(rc_roi, 'RGB')
        rc_image = cv2.resize(rc_roi, (cheek_frame_size, cheek_frame_size))
        prev_rc_image = rc_image
    
    # Update the last 30 seconds deque
    lc_last30seconds.append(lc_red_intensity)
    rc_last30seconds.append(rc_red_intensity)  
    frame_last30seconds.append(red_intensity_frame) 
    lrc_last30seconds.append(lrc_red_intensity)

    if runningTime[0] > 0:
        runningWindow_time_elapsed = runningTime[-1] - runningTime[0]
        # Perform FFT on the last 30 seconds
        lc_fft    = fft_on_deque(lc_last30seconds, runningWindow_time_elapsed)
        rc_fft    = fft_on_deque(rc_last30seconds, runningWindow_time_elapsed)
        frame_fft = fft_on_deque(frame_last30seconds, runningWindow_time_elapsed)
        lrc_fft   = fft_on_deque(lrc_last30seconds, runningWindow_time_elapsed)

        # Update the bpm deques
        lc_bpm.append(lc_fft * 60)
        rc_bpm.append(rc_fft * 60)
        frame_bpm.append(frame_fft * 60)
        lrc_bpm.append(lrc_fft * 60)

        # Calculate the average bpm
        lc_avg_bpm    = np.mean(lc_bpm)
        rc_avg_bpm    = np.mean(rc_bpm)
        frame_avg_bpm = np.mean(frame_bpm)
        lrc_avg_bpm   = np.mean(lrc_bpm)

        # Display the max fft frequency on the frame
        cv2.putText(frame, f"LC BPM: {lc_avg_bpm:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"RC BPM: {rc_avg_bpm:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame BPM: {frame_avg_bpm:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"LRC BPM: {lrc_avg_bpm:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Update the red intensity array
    red_intensity_over_time.append(red_intensity_frame)
    lc_red_intensity_over_time.append(lc_red_intensity)
    rc_red_intensity_over_time.append(rc_red_intensity)
    lrc_red_intensity_over_time.append(lrc_red_intensity)

    # # Update the last 30 seconds deque
    # lc_last30seconds.append(lc_red_intensity)
    # rc_last30seconds.append(rc_red_intensity)  
    # frame_last30seconds.append(red_intensity_frame) 
    # lrc_last30seconds.append(lrc_red_intensity)

    # Show the image and write to video
    cv2.imshow(f"Left Cheek", prev_lc_image)
    cv2.imshow(f"Right Cheek", prev_rc_image)
    cv2.imshow("Facial Landmarks", frame)
    if DO_WRITE_VIDEO:
        lc_vid_writer.write(prev_lc_image)
        rc_vid_writer.write(prev_rc_image)
        face_vid_writer.write(frame_clean)

    end_of_frame_time = time.time()
    # print(f"Frame Time: {end_of_frame_time - start_of_frame_time}")

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        time_end = time.time()
        break

time_x = np.arange(0, num_frames) / fps

tot_time = time_end - time_start
fps = num_frames / tot_time
print(f"Frames: {num_frames}, Time: {tot_time}, FPS: {fps}")

# plot red intensity over time and red intensity over last 30 seconds all in one plot using subplots
fig, axs = plt.subplots(2, 2)
fig.suptitle('Red Intensity Over Time')
axs[0, 0].plot(time_x, red_intensity_over_time)
axs[0, 0].set_title('Whole Face')
axs[0, 1].plot(time_x, lc_red_intensity_over_time)
axs[0, 1].set_title('Left Cheek')
axs[1, 0].plot(time_x, rc_red_intensity_over_time)
axs[1, 0].set_title('Right Cheek')
axs[1, 1].plot(time_x, lrc_red_intensity_over_time)
axs[1, 1].set_title('Left-Right Cheek')
plt.show()

# Save the arrays as a npz file
np.savez("red_intensity_over_time", time=time_x, intensity=red_intensity_over_time)
np.savez("lc_red_intensity_over_time", time=time_x, intensity=lc_red_intensity_over_time)
np.savez("rc_red_intensity_over_time", time=time_x, intensity=rc_red_intensity_over_time)
np.savez("lrc_red_intensity_over_time", time=time_x, intensity=lrc_red_intensity_over_time)

# # Call peak analysis function
# peak_analysis("red_intensity_over_time.npz", "", 0, 1)
# peak_analysis("lc_red_intensity_over_time.npz", "", 0, 1)
# peak_analysis("rc_red_intensity_over_time.npz", "", 0, 1)
# peak_analysis("lrc_red_intensity_over_time.npz", "", 0, 1)
print(f"Time elapsed: {num_frames/fps} seconds")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
