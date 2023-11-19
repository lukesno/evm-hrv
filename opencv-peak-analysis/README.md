# open-cv-peak-analysis

# command line
To run program, use
`python peak_analysis.py [filename] --height [value] --distance [value]`
Example:
```python peak_analysis.py ../opencv-head-detector luke_face_after_EVM_after_red_channel_extraction_red_intensity.npy --height 0.8 --distance 20```
To close program, close the last plot.

# peak detection parameters
height: sets the minimum height of peaks that will be detected. Only peaks with values greater than or equal to this threshold will be considered.
distance: sets the minimum horizontal distance between peaks. It ensures that only the tallest peaks within a specified distance range are detected

# 1) FFT (Frequency Domain)
The Fast Fourier Transform (FFT) is a mathematical algorithm that transforms a signal from its original time domain to its frequency domain representation. In the context of Heart Rate Variability (HRV), FFT is used to analyze the frequency components present in the inter-beat interval time series, revealing insights into autonomic nervous system activity. The resulting frequency spectrum, divided into bands such as Very Low Frequency (VLF), Low Frequency (LF), and High Frequency (HF), allows extraction of HRV features indicative of physiological processes and autonomic control [1] [2].

HRV analysis often focuses on different frequency bands, namely:
Very Low Frequency (VLF): < 0.04 Hz [3]
Low Frequency (LF): 0.04 - 0.15 Hz [4]
High Frequency (HF): 0.15 - 0.4 Hz [5]

References:

1. Malik, M. (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. European Heart Journal, 17(3), 354–381.
2. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.
3. Berntson, G. G., Bigger, J. T., Jr, Eckberg, D. L., Grossman, P., Kaufmann, P. G., Malik, M., ... & Van Der Molen, M. W. (1997). Heart rate variability: origins, methods, and interpretive caveats. Psychophysiology, 34(6), 623-648.
4. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.
5. Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in public health, 5, 258.

# 2) Non-Linear Method
TODO

# 3) Time-Domain Method
TODO

# future wishes/issues
- There are huge spikes in the FFT graph that may be outliers -> make it hard to see rest of data and could mess with analysis?
- There is non-linear method and frequency domain method that need to be cooked


