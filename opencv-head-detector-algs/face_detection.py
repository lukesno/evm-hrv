import cv2
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from the webcam
cap = cv2.VideoCapture(0)

# Time duration to keep the face frame fixed
fixed_frame_duration = 5 * 60  # 5 minutes converted to seconds

cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap_fps = cap.get(cv2.CAP_PROP_FPS)
fps_sleep = int(1000 / cap_fps)
print('* Capture width:', cap_width)
print('* Capture height:', cap_height)
print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')


# Set up the video capture width and height for compatibility
# cap.set(cv2.CV_CAP_PROP_FPS, 30)
cap.set(3, 1280)  # Set width as 640
cap.set(4, 720)  # Set height as 480

# Initialize variables to store the coordinates of the face
face_coords = None
start_time = None

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
actual_fps = cap.get(30)
print(actual_fps)
out = None
# out = cv2.VideoWriter('face_output.mp4', fourcc, 15.0, (y+h, x+w))
cropped_dimensions = (0, 0)

while True:
    # Read the frame
    ret, img = cap.read()
    
    # If frame reading was not successful, skip this iteration
    if not ret:
        continue

    if face_coords is None or (time.time() - start_time) > fixed_frame_duration:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)

        # Check if we found a face, if so, update the face coordinates and reset the timer
        if len(faces) > 0:
            face_coords = faces[0]  # (x, y, w, h)
            x, y, w, h = face_coords
            face_roi = img[y:y+h, x:x+w]
            height, width = face_roi.shape[:2]
            out = cv2.VideoWriter('face_output.mp4', fourcc, 15.0, (height, width))
            start_time = time.time()

    if face_coords is not None:
        # Use the stored coordinates to keep the same face frame
        x, y, w, h = face_coords
        face_roi = img[y:y+h, x:x+w]
        height, width = face_roi.shape[:2]

        face_roi_resized = cv2.resize(face_roi, (height, width))

        out.write(face_roi_resized)

        cv2.imshow('Face Detection - Cropped', face_roi)
    else:
        # If no face coordinates stored yet, just show the full frame
        cv2.imshow('Face Detection', img)

    # Stop if the escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break

# Release the VideoCapture object and close display window
cap.release()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()
