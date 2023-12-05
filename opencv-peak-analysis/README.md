# open-cv-peak-analysis

# command line
To run program, use
`python peak_analysis.py [filename] --height [value] --distance [value]`
Example:
```python peak_analysis.py ../head-detector-opencv/luke_face_after_EVM_after_red_channel_extraction_red_intensity.npy --height 0.8 --distance 20```
`python peak_analysis.py ../opencv-red-channel-extraction/output_red_channel_red_intensity.npz --height 0.8 --distance 20`

To close program, close the last plot.

# peak detection parameters
height: sets the minimum height of peaks that will be detected. Only peaks with values greater than or equal to this threshold will be considered.
distance: sets the minimum horizontal distance between peaks. It ensures that only the tallest peaks within a specified distance range are detected


# future wishes/issues



