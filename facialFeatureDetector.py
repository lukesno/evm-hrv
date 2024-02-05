import cv2
import dlib
from PIL import Image
import numpy as np

# Load the pre-trained face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the facial landmark predictor from dlib
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
cap = cv2.VideoCapture(0)
# Get video details
# fps    = int(cap.get(cv2.CAP_PROP_FPS))
fps    = 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to save the output
# fourcc = cv2.VideoWriter_fourcc(*'H264')
fourcc          = cv2.VideoWriter_fourcc(*'avc1')
face_vid_writer = cv2.VideoWriter("whole_face_output.mp4", fourcc, fps, (width, height), isColor=True)
lc_vid_writer   = cv2.VideoWriter("left_cheek_output.mp4", fourcc, fps, (cheek_frame_size, cheek_frame_size), isColor=True)
rc_vid_writer   = cv2.VideoWriter("right_cheek_output.mp4", fourcc, fps, (cheek_frame_size, cheek_frame_size), isColor=True)

# Create an array to store red intensity over time
red_intensity_over_time = []
lc_red_intensity_over_time = []
rc_red_intensity_over_time = []

blank_frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # Initialize with white pixels (for RGB images)
blank_image = Image.fromarray(blank_frame, 'RGB')
blank_image = cv2.resize(blank_frame, (cheek_frame_size, cheek_frame_size))
prev_rc_roi = blank_frame
prev_lc_roi = blank_frame

num_frames = 0

while True:
    # Capture frame-by-frame
    ret, frame_clean = cap.read()
    frame = frame_clean.copy()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # gray = frame[:,:,2]

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
        
        LC_top   += (LC_bot - LC_top)//4
        LC_left  -= (LC_right - LC_left)//4
        LC_right -= (LC_right - LC_left)//10

        # Calculate pixels corresponding to right cheek
        RC_top   = landmarks.part(45).y
        RC_bot   = landmarks.part(35).y
        RC_left  = landmarks.part(35).x
        RC_right = landmarks.part(45).x 

        RC_top   += (RC_bot - RC_top)//4
        RC_right += (RC_right - RC_left)//4
        RC_left  += (RC_right - RC_left)//10

        # Draw landmarks on the face
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # Draw rectangles around cheeks
        cv2.rectangle(frame, (LC_left, LC_top), (LC_right, LC_bot), (0, 0, 255), 2)
        cv2.rectangle(frame, (RC_left, RC_top), (RC_right, RC_bot), (0, 0, 255), 2)
        
        lc_roi = frame_clean[LC_top:LC_bot, LC_left:LC_right]
        rc_roi = frame_clean[RC_top:RC_bot, RC_left:RC_right]

        # Calculate the average red intensity of the frame ALREADY RED OR IR
        red_intensity_frame = np.mean(frame)
        lc_red_intensity    = np.mean(lc_roi)
        rc_red_intensity    = np.mean(rc_roi)

        # Update the red intensity array
        red_intensity_over_time.append(red_intensity_frame)
        lc_red_intensity_over_time.append(lc_red_intensity)
        rc_red_intensity_over_time.append(rc_red_intensity)
        
        # Ensure that both images have the same height
        rc_height  = RC_bot - RC_top
        lc_height  = RC_bot - LC_top 
        rc_width   = RC_right - RC_left
        lc_width   = LC_right - LC_left

        # max_height = max(lc_height, rc_height)
        # lc_roi     = np.resize(lc_roi, (max_height, int(lc_width * (max_height / lc_height))))
        # rc_roi     = np.resize(rc_roi, (max_height, int(rc_width* (max_height / rc_height))))


        # combined_cheeks = np.concatenate((lc_roi, rc_roi), axis=1)
        # # Combine the images horizontally
        # combined_image = Image.new('RGB', (lc_width + rc_width, max_height))
        # combined_image.paste(lc_roi, (0, 0))
        # combined_image.paste(rc_roi, (lc_width, 0)) 

        # cv2.imshow("Cheeks", combined_cheeks)
        if lc_roi.size:
            lc_image = Image.fromarray(lc_roi, 'RGB')
            lc_image = cv2.resize(lc_roi, (cheek_frame_size, cheek_frame_size))
            prev_lc_image = lc_roi
            cv2.imshow(f"Left Cheek", lc_image)
            lc_vid_writer.write(lc_image)
        else:
            lc_vid_writer.write(prev_lc_image)

        if rc_roi.size:
            rc_image = Image.fromarray(rc_roi, 'RGB')
            rc_image = cv2.resize(rc_roi, (cheek_frame_size, cheek_frame_size))
            prev_rc_image = rc_image
            cv2.imshow(f"Right Cheek", rc_image)
            rc_vid_writer.write(rc_image)
        else:
            lc_vid_writer.write(prev_rc_image)

        # One face only
        break

    # Display the frame
    cv2.imshow("Facial Landmarks", frame)
    face_vid_writer.write(frame_clean)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time = np.arange(0, num_frames) / fps
np.savez("whole_face_output.npy", time=time, red_intensity=red_intensity_over_time)
np.savez("left_cheek_output.npy", time=time, red_intensity=red_intensity_over_time)
np.savez("right_cheek_output.npy", time=time, red_intensity=red_intensity_over_time)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()