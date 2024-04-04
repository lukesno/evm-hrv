#===========================================================
# Import necessary libraries
import cv2
import dlib
from PIL import Image
import numpy as np
import os
import time
#===========================================================

#===========================================================
# Constants
cheek_frame_size = 300
DO_WRITE_VIDEO = True 
DO_SHOW_VIDEO  = True 
FACIAL_PIXEL_DEVIATION_THRESHOLD = 15
FRAMES_PER_FACIAL_DETECTION = 10 

#===========================================================

#===========================================================
# Set up
# Load the pre-trained face detector from dlib
# Load the facial landmark predictor from dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open the camera
src = 1
cap = cv2.VideoCapture(src)
fps    = 30
# Check if camera or videofile (src is integer or string)
if isinstance(src, int):
    is_camera = True
else:
    is_camera = False
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

print(f"FPS: {fps}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a new direct named 'out' under vids, if 'out' does not exist
if not os.path.exists("vids/out"):
    os.makedirs("vids/out")

# Create VideoWriter object to save the output
if DO_WRITE_VIDEO:
    fourcc          = cv2.VideoWriter_fourcc(*'mp4v')
    lc_vid_writer   = cv2.VideoWriter("vids/out/left_cheek_output.mp4", fourcc, fps, (cheek_frame_size, cheek_frame_size), isColor=True)
    rc_vid_writer   = cv2.VideoWriter("vids/out/right_cheek_output.mp4", fourcc, fps, (cheek_frame_size, cheek_frame_size), isColor=True)
#===========================================================

#===========================================================
# Init Variables 
print(f"Video Details/: FPS: {fps}, Width: {width}, Height: {height}")
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

num_frames = 0
winSize             = 100
#================================================================
   
#================================================================
# MAIN
#================================================================
# start measuing the time of the program
time_start                       = time.time()    
frames_since_last_face_detection = FRAMES_PER_FACIAL_DETECTION

while True:
    # Capture frame-by-frame
    ret, frame    = cap.read()
    # frame_clean = cv2.rotate(frame_clean, cv2.ROTATE_180)
    if ret == False:
        break
    if is_camera:
        start_of_frame_time = time.time()
    else:
        start_of_frame_time = 1/fps * num_frames
    num_frames += 1

    if frames_since_last_face_detection == FRAMES_PER_FACIAL_DETECTION:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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

            break

        frames_since_last_face_detection = 0
    else:
        frames_since_last_face_detection += 1
    
    lc_roi = frame[LC_top_cur:LC_bot_cur, LC_left_cur:LC_right_cur]
    rc_roi = frame[RC_top_cur:RC_bot_cur, RC_left_cur:RC_right_cur]

    # Update images if they are not empty
    if lc_roi.size:
        lc_image = Image.fromarray(lc_roi, 'RGB')
        lc_image = cv2.resize(lc_roi, (cheek_frame_size, cheek_frame_size))
        prev_lc_image = lc_image
    if rc_roi.size:
        rc_image = Image.fromarray(rc_roi, 'RGB')
        rc_image = cv2.resize(rc_roi, (cheek_frame_size, cheek_frame_size))
        prev_rc_image = rc_image

    # Show the image and write to video
    if DO_SHOW_VIDEO:
        cv2.imshow(f"Left Cheek", prev_lc_image)
        cv2.imshow(f"Right Cheek", prev_rc_image)
        cv2.imshow("Facial Landmarks", frame)

    if DO_WRITE_VIDEO:
        lc_vid_writer.write(prev_lc_image)
        rc_vid_writer.write(prev_rc_image)

    end_of_frame_time = time.time()
    # print(f"Frame Time: {end_of_frame_time - start_of_frame_time}")

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time_end = time.time()
time_x = np.arange(0, num_frames) / fps

tot_time = time_end - time_start
fps = num_frames / tot_time
print(f"Frames: {num_frames}, Time: {tot_time}, FPS: {fps}")
print(f"Time elapsed: {num_frames/fps} seconds")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
