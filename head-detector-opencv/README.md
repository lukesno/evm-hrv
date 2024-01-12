# head-detector-opencv
Basic scripts that output a video of a detected human head in your webcam


face_detection.py -> crops screen to first detected head, stays on it for 10 minutes\
auto_lock_detection.py -> crops screen dynamically to detected head after 33ms. May be more inaccurate depending on environment
\
\
\
Prereqs:
Python3, opencv\
Run ```pip install opencv-python``` in your terminal before running!
\
\
To run program, use ```python auto_lock_detection.py```\
To close program, click esc

# future wishes/issues
Only make it detect for the amount of time we are collecting data for which is 10-15 minutes. For now we close the window manually to stope recording.

