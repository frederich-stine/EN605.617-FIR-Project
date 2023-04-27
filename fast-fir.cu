// Frederich Stine EN.605.617
// Module 8 Assignment Part 2

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <chrono>

#include <cufft.h>
#include <fftw3.h>

#include "filter_parse.h"
#include "wav_parse.h"

/******************* CUDA Kernel Prototypes ********************/
__global__ 
void gpuFIRKernel(double* filter, double* audio, 
	double* audioOut, int filterLen, int workSize);
__global__
void gpuNormalize(double* audio, double maxValue, int size);
__global__
void gpuMagnitude(cufftDoubleComplex* fftResult, double* output, int resultSize);
__global__
void gpuSum(double* input, double* output, double run, int resultSize);
__global__
void gpuDiv(double* input, double run, int resultSize);
__global__
void gpuDB(double* output, int resultSize);


/******************* Core Function Prototypes ********************/
// Function to run an FFT on a simple audio file
// void runFFT (void);
void cpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut);
void gpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut);
void gpuFFT(wavParse* audio, int fftSize, char* fileName);
void cpuFFT(wavParse* audio, int fftSize, char* fileName);

/******************* Helper Function Prototypes ********************/
void printFFTResults(double* buffer, int fftSize, int sampleRate);
void writeFFTResults(double* buffer, int fftSize, int sampleRate, char* fileName);

/******************* Global Variables ********************/

/******************* Funtion definitions ********************/
int main (int argc, char** argv) {
	// Prints out a help menu if not enough params are passed
	if (argc != 5) {
		printf("Error: Correct usage is\n");
		printf("     : ./fast-fir filter audioFileIn audioFileOut {operation}\n");
		printf("     : operation :\n");
		printf("                 : 0 - CPU\n");
		printf("                 : 1 - GPU\n");
		exit(0);
	}

	filterParse* filter = new filterParse(argv[1]);
	wavParse* audio = new wavParse(argv[2]);
	wavWrite* audioOut = new wavWrite(argv[3]);
	audioOut->writeHeader(&audio->header);

	/*
	printf("CoeffArrValue: %lf\n", filter->coeffArr[filter->len-1]);
	printf("FilterLen: %d\n", filter->len);
	printf("NumChannels: %d\n", audio->header.NumChannels);
	printf("SampleRate: %d\n", audio->header.SampleRate);
	printf("BitsPerSample: %d\n", audio->header.BitsPerSample);
	*/

	audio->workSize = 20000;

	int operation = atoi(argv[4]);

	switch (operation) {
		case 0:
			cpuFIR(filter, audio, audioOut);
			break;
		case 1:
			gpuFIR(filter, audio, audioOut);
			break;
	}

	delete(audioOut);
	wavParse* filteredAudio = new wavParse(argv[3]);

	switch (operation) {
		case 0:
			cpuFFT(audio, 2048, "FFT_Results_CPU_noFilter.txt");
			cpuFFT(filteredAudio, 2048, "FFT_Results_CPU_Filter.txt");
			break;
		case 1:
			gpuFFT(audio, 2048, "FFT_Results_GPU_noFilter.txt");
			gpuFFT(filteredAudio, 2048, "FFT_Results_GPU_Filter.txt");
			break;
	}

	delete(filteredAudio);
	delete(filter);
	delete(audio);
}

void cpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut) {
	
	std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

	size_t workSize = audio->loadWorkSize(filter->len);

	while (workSize > filter->len) {

		double temp = 0;
		for (int x=0; x<(workSize-filter->len); x++) {
			temp = 0;
			for (int i=x; i<(filter->len+x); i++) {
				temp += audio->audioBuf[i] * filter->coeffArr[i-x];
			}
			audioOut->writeSample(temp);
		}

		workSize = audio->loadWorkSize(filter->len);
	}

	// https://gist.github.com/mortie/bf21c9d2d53b83f3be1b45b76845f090
	std::chrono::time_point stopTime = std::chrono::high_resolution_clock::now();
	auto ns = std::chrono::duration<double>(stopTime - startTime);
	printf("CPU FIR Runtime: %lfs\n", ns.count());
}

void gpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut) {
	double *audioOutBufHost;
	double *filterBuf, *audioBuf, *audioOutBuf;

	cudaMallocHost((void**)&audioOutBufHost, audio->workSize*(sizeof(double)));

	cudaMalloc((void**)&filterBuf, filter->len*(sizeof(double)));
	cudaMalloc((void**)&audioBuf, audio->workSize*(sizeof(double)));
	cudaMalloc((void**)&audioOutBuf, audio->workSize*(sizeof(double)));

	cudaMemcpy(filterBuf, filter->coeffArr, filter->len*(sizeof(double)), cudaMemcpyHostToDevice);

	std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();
	size_t workSize = audio->loadWorkSize(filter->len);

	while (workSize > filter->len) {
		int batchSize = workSize - filter->len;
		int blockSize = 64;
		int numBlocks = batchSize+(blockSize-1) / blockSize;

		cudaMemcpy(audioBuf, audio->audioBuf, workSize*(sizeof(double)), cudaMemcpyHostToDevice);

		gpuFIRKernel<<<numBlocks, blockSize>>>(filterBuf, audioBuf, audioOutBuf, filter->len, workSize);

		cudaMemcpy(audioOutBufHost, audioOutBuf, audio->workSize*(sizeof(double)), cudaMemcpyDeviceToHost);

		audioOut->writeBulk(audioOutBufHost, workSize-filter->len);

		workSize = audio->loadWorkSize(filter->len);
	}

	// https://gist.github.com/mortie/bf21c9d2d53b83f3be1b45b76845f090
	std::chrono::time_point stopTime = std::chrono::high_resolution_clock::now();
	auto ns = std::chrono::duration<double>(stopTime - startTime);
	printf("GPU FIR Runtime: %lfs\n", ns.count());

	cudaFree(filterBuf);
	cudaFree(audioBuf);
	cudaFree(audioOutBuf);

	cudaFreeHost(audioOutBufHost);
}

void gpuFFT(wavParse* audio, int fftSize, char* fileName) {
	audio->reset();

	int resultSize = (fftSize/2)+1;
	int resultSizeBytes = resultSize*sizeof(cufftDoubleComplex);
	int resultSizeMag = resultSize*sizeof(double);

	int blockSize = 64;
	int numBlocks = (fftSize+blockSize-1)/blockSize;
	int numBlocksResult = (resultSize+blockSize-1)/blockSize;

	cufftDoubleComplex* o_cu_buf;
	cudaMallocHost((void**)&o_cu_buf, \
			fftSize*sizeof(cufftDoubleComplex));

	cufftDoubleReal* d_i_cu_buf;
	cufftDoubleComplex* d_o_cu_buf;
	double* d_o_mag_buf;
	double* d_o_mag_avg;

	cudaMalloc((void **)&d_i_cu_buf, \
			fftSize*sizeof(cufftDoubleReal));
	cudaMalloc((void **)&d_o_cu_buf, resultSizeBytes);
	cudaMalloc((void **)&d_o_mag_buf, resultSizeMag);
	cudaMalloc((void **)&d_o_mag_avg, resultSizeMag);

	cufftHandle plan;
	cufftPlan1d(&plan, fftSize, CUFFT_D2Z, 1);

	// Result buffer
	double* d_o_host_buf = (double*) malloc (resultSize*sizeof(double));
	double* d_o_host_buf_avg = (double*) malloc (resultSize*sizeof(double));

	for (int i=0; i<resultSize; i++) {
		d_o_host_buf_avg[i] = 0.0;
	}

	std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

	size_t bytesRead = audio->loadData(fftSize);

	float maxValue = ((pow(2, (float)audio->header.BitsPerSample))/2)-1.0;

	double count = 1.0;
	while (bytesRead == fftSize) {
		cudaMemcpy(d_i_cu_buf, audio->audioBuf, \
				fftSize*sizeof(cufftDoubleReal),\
				cudaMemcpyHostToDevice);

		gpuNormalize<<<numBlocks, blockSize>>>(d_i_cu_buf, maxValue, fftSize);

		// Execute the fft
		cufftExecD2Z(plan, d_i_cu_buf, d_o_cu_buf);

		gpuMagnitude<<<numBlocksResult, blockSize>>>(d_o_cu_buf, d_o_mag_buf, resultSize);

		gpuSum<<<numBlocksResult, blockSize>>>(d_o_mag_avg, d_o_mag_buf, count, resultSize);

		bytesRead = audio->loadData(fftSize);
		count++;
	}

	gpuDiv<<<numBlocksResult, blockSize>>>(d_o_mag_avg, count, resultSize);
	gpuDB<<<numBlocksResult, blockSize>>>(d_o_mag_avg, resultSize);

	cudaMemcpy(d_o_host_buf_avg, d_o_mag_avg, \
		resultSizeMag, cudaMemcpyDeviceToHost);

	// https://gist.github.com/mortie/bf21c9d2d53b83f3be1b45b76845f090
	std::chrono::time_point stopTime = std::chrono::high_resolution_clock::now();
	auto ns = std::chrono::duration<double>(stopTime - startTime);
	printf("GPU FFT Runtime: %lfs\n", ns.count());

	//printFFTResults(d_o_host_buf_avg, fftSize, audio->header.SampleRate);
	writeFFTResults(d_o_host_buf_avg, fftSize, audio->header.SampleRate, fileName);

	free(d_o_host_buf);
	free(d_o_host_buf_avg);

	// Free all memory
	cufftDestroy(plan);

	cudaFree(d_i_cu_buf);
	cudaFree(d_o_cu_buf);
	cudaFree(d_o_mag_buf);
	cudaFree(d_o_mag_avg);

	cudaFreeHost(o_cu_buf);
}

void cpuFFT(wavParse* audio, int fftSize, char* fileName) {
	audio->reset();

	int resultSize = (fftSize/2)+1;
	fftw_complex* output;
	fftw_plan plan;
	double* magnitude = (double*) malloc(resultSize*sizeof(double));
	double* magnitudeAvg = (double*) malloc(resultSize*sizeof(double));
	double* magnitudeDbAvg = (double*) malloc(resultSize*sizeof(double));
	output = (fftw_complex*) malloc(resultSize*sizeof(fftw_complex));
	float maxValue = ((pow(2, (float)audio->header.BitsPerSample))/2)-1.0;

	for (int i=0; i<resultSize; i++) {
		magnitudeAvg[i] = 0.0;
	}

	plan = fftw_plan_dft_r2c_1d(fftSize, audio->audioBuf, output, FFTW_ESTIMATE);

	std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

	size_t bytesRead = audio->loadData(fftSize);

	double run = 1.0;
	//input = (fftw_complex*) malloc(fftSize*sizeof(fftw_complex));
	while (bytesRead == fftSize) {

		// Normalize data
		for (int i=0; i<fftSize; i++) {
			audio->audioBuf[i] = audio->audioBuf[i]/maxValue;
		}
		
		plan = fftw_plan_dft_r2c_1d(fftSize, audio->audioBuf, output, FFTW_ESTIMATE);
		fftw_execute(plan);

		for (int i=0; i<resultSize; i++) {
			magnitude[i] = sqrt(pow(output[i][0], 2) + pow(output[i][1], 2));
		}

		for (int i=0; i<resultSize; i++) {
			magnitudeAvg[i] += magnitude[i];
		}		

		bytesRead = audio->loadData(fftSize);
		run++;
	}

	for (int i=0; i<resultSize; i++) {
		magnitudeAvg[i] = magnitudeAvg[i]/run;
		magnitudeDbAvg[i] = 20*log10(magnitudeAvg[i]);
	}

	// https://gist.github.com/mortie/bf21c9d2d53b83f3be1b45b76845f090
	std::chrono::time_point stopTime = std::chrono::high_resolution_clock::now();
	auto ns = std::chrono::duration<double>(stopTime - startTime);
	printf("CPU FFT Runtime: %lfs\n", ns.count());

	//printFFTResults(magnitudeDbAvg, fftSize, audio->header.SampleRate);
	writeFFTResults(magnitudeDbAvg, fftSize, audio->header.SampleRate, fileName);

	free(magnitudeDbAvg);
	free(magnitudeAvg);
	free(magnitude);
	free(output);

}

__global__ 
void gpuFIRKernel(double* filter, double* audio, 
	double* audioOut, int filterLen, int workSize) {
	
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_idx >= workSize-filterLen) {
		return;
	}

	double temp = 0;

	for (int i=0; i<filterLen; i++) {
		temp += audio[i+thread_idx] * filter[i];
	}

	audioOut[thread_idx] = temp;
}

__global__
void gpuNormalize(double* audio, double maxValue, int size) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_idx >= size) {
		return;
	}

	audio[thread_idx] = audio[thread_idx]/maxValue;
}

__global__
void gpuMagnitude(cufftDoubleComplex* fftResult, double* output, int resultSize) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (thread_idx >= resultSize) {
		return;
	}

	output[thread_idx] = sqrt(pow(fftResult[thread_idx].x, 2) + pow(fftResult[thread_idx].y, 2));

	//output[thread_idx] = 20*log10(output[thread_idx]);
}

__global__
void gpuSum(double* input, double* output, double run, int resultSize) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_idx >= resultSize) {
		return;
	}

	if (run == 1.0) {
		input[thread_idx] = output[thread_idx];
	}
	else {
		input[thread_idx] += output[thread_idx];
	}
}

__global__
void gpuDiv(double* input, double run, int resultSize) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_idx >= resultSize) {
		return;
	}

	input[thread_idx] = input[thread_idx]/run;
}

__global__
void gpuDB(double* output, int resultSize) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (thread_idx >= resultSize) {
		return;
	}

	output[thread_idx] = 20*log10(output[thread_idx]);
}

void printFFTResults(double* buffer, int fftSize, int sampleRate) {
	for (int i=0; i<(fftSize/2)+1; i++) {
		float frequency = (float)i*(float)sampleRate/(float)fftSize;
		printf("FFT Result: Frequency: %f: Magnitude dB: %f\n", \
				frequency, buffer[i]);
	}
}

void writeFFTResults(double* buffer, int fftSize, int sampleRate, char* fileName) {
	FILE* fh = fopen(fileName, "w");
	if (fh==NULL) {
		printf("Error: Failed to write FFT results\n");
		exit(0);
	}

	for (int i=0; i<(fftSize/2)+1; i++) {
		float frequency = (float)i*(float)sampleRate/(float)fftSize;
		fprintf(fh, "FFT Result: Frequency: %f: Magnitude dB: %f\n", \
				frequency, buffer[i]);
	}

	fclose(fh);
}