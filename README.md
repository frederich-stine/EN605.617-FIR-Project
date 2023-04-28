# EN605.617-FIR-Project

This is the FIR FFT project for EN605.617 JHU SP 23

This project takes in 16-bit signed mono wav files and performs FIR filtering and FFT calculation using either the CPU or GPU. The
filters can be created in PyFDA (included as a submodule) and a loaded into the program. FFTs are performed using cuFFT and FFTW3.

This program compares the performance of these calculations for varying sizes on the CPU and GPU.

## Dependencies

- PyFDA
- Cuda
- cuFFT
- FFTW3

FFTW3 will have to be downloaded and installed on your machine.

## Example Results

Full results from the included *run.sh* example script can be viewed at [documentation/full_run.log](documentation/full_run.log).  

```
CPU FIR and FFT - 440Hz 5Sec - FFT Size 1024 - bandstop
----- Wav Info -----
ChunkSize: 444516
SubChunkSize: 16
AudioFormat: 1
NumChannels: 1
SampleRate: 44100
ByteRate: 88200
BlockAlign: 2
BitsPerSample: 16
SubChunk2Size 16

CPU FIR Runtime: 1.297875s
CPU FFT Runtime: 0.008923s
CPU FFT Runtime: 0.008430s
Press enter to continue:
GPU FIR and FFT - 440Hz 5Sec - FFT Size 1024 - bandstop
----- Wav Info -----
ChunkSize: 444516
SubChunkSize: 16
AudioFormat: 1
NumChannels: 1
SampleRate: 44100
ByteRate: 88200
BlockAlign: 2
BitsPerSample: 16
SubChunk2Size 16

GPU FIR Runtime: 0.009123s
GPU FFT Runtime: 0.005477s
GPU FFT Runtime: 0.005185s
```

From these results we can see that the GPU is significantly faster than the CPU at performing the FIR convolution. The performance
difference is not as large when performing FFT calculations. This is partially due to not performing batch operations in cuFFT, but when profiled I found that only a small percentage of time was spent copying memory during the cuFFT portion. FFTW3 is a very well
optimized library and even though this project uses it single threaded it utilizes the vector extensions of the CPU to increase
performance.  

## Areas for Future Improvement

This project meets all of the goals I set out to complete. It could be expanded in the future with more development.

- Stereo wav files
- Different sample sizes
- cuFFT batch operations