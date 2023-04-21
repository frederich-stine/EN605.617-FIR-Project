// Frederich Stine EN.605.617
// Module 8 Assignment Part 2

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include <cufft.h>

#include "filter_parse.h"
#include "wav_parse.h"

/******************* CUDA Kernel Prototypes ********************/
__global__ 
void gpuFIRKernel(double* filter, double* audio, 
	double* audioOut, int filterLen, int workSize);

/******************* Core Function Prototypes ********************/
// Function to run an FFT on a simple audio file
// void runFFT (void);
void cpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut);
void gpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut);
void gpuFFT(wavParse* audio);

/******************* Helper Function Prototypes ********************/

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

	printf("CoeffArrValue: %lf\n", filter->coeffArr[filter->len-1]);
	printf("FilterLen: %d\n", filter->len);
	printf("NumChannels: %d\n", audio->header.NumChannels);
	printf("SampleRate: %d\n", audio->header.SampleRate);
	printf("BitsPerSample: %d\n", audio->header.BitsPerSample);

	audio->workSize = 10000;

	int operation = atoi(argv[4]);

	printf("FIR filtering\n");
	switch (operation) {
		case 0:
			cpuFIR(filter, audio, audioOut);
		case 1:
			gpuFIR(filter, audio, audioOut);
	}

	delete(audioOut);

	wavParse* filteredAudio = new wavParse(argv[3]);

	//filteredAudio->loadData(100);
	//filteredAudio->loadWorkSize(500);
	//filteredAudio->loadData(500);

	gpuFFT(audio);
	gpuFFT(filteredAudio);

	delete(filteredAudio);
	delete(filter);
	delete(audio);

	//printf("FFT calculation\n");
	//gpuFFT(&audio);
	//audioOut.reset();
	//audioOut.loadData(500);

	//gpuFFT(&audioOut);

	// Run core function
	//runFFT();
}

void cpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut) {
	
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
}

void gpuFIR(filterParse* filter, wavParse* audio, wavWrite* audioOut) {
	double *audioOutBufHost;
	double *filterBuf, *audioBuf, *audioOutBuf;

	cudaMallocHost((void**)&audioOutBufHost, audio->workSize*(sizeof(double)));

	cudaMalloc((void**)&filterBuf, filter->len*(sizeof(double)));
	cudaMalloc((void**)&audioBuf, audio->workSize*(sizeof(double)));
	cudaMalloc((void**)&audioOutBuf, audio->workSize*(sizeof(double)));

	cudaMemcpy(filterBuf, filter->coeffArr, filter->len*(sizeof(double)), cudaMemcpyHostToDevice);

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

	cudaFree(filterBuf);
	cudaFree(audioBuf);
	cudaFree(audioOutBuf);

	cudaFreeHost(audioOutBufHost);
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

void gpuFFT(wavParse* audio) {
	audio->reset();

	int numSamples = 2048;
	int resultSize = (numSamples/2)+1;
	int resultSizeBytes = resultSize*sizeof(cufftDoubleComplex);

	cufftDoubleComplex* o_cu_buf;
	cudaMallocHost((void**)&o_cu_buf, \
			numSamples*sizeof(cufftDoubleComplex));

	cufftDoubleReal* d_i_cu_buf;
	cufftDoubleComplex* d_o_cu_buf;

	cudaMalloc((void **)&d_i_cu_buf, \
			numSamples*sizeof(cufftDoubleReal));
	cudaMalloc((void **)&d_o_cu_buf, resultSizeBytes);

	cufftHandle plan;
	cufftPlan1d(&plan, numSamples, CUFFT_D2Z, 1);

	size_t bytesRead = audio->loadData(numSamples);

	while (bytesRead == numSamples) {

		cudaMemcpy(d_i_cu_buf, audio->audioBuf, \
				numSamples*sizeof(cufftDoubleReal),\
				cudaMemcpyHostToDevice);

		// Execute the fft
		cufftExecD2Z(plan, d_i_cu_buf, d_o_cu_buf);

		// Copy result back from GPU
		cudaMemcpy(o_cu_buf, d_o_cu_buf, \
				resultSizeBytes, cudaMemcpyDeviceToHost);

		float* d_o_magnitude = (float*) malloc (resultSize*sizeof(float));
		for (int i=0; i<resultSize; i++) {
			d_o_magnitude[i] = sqrt(pow(o_cu_buf[i].x, 2) + pow(o_cu_buf[i].y, 2));
		}

		// Calculate magnitude db
		float* d_o_magnitude_db = (float*) malloc (resultSize*sizeof(float));
		for (int i=0; i<resultSize; i++) {
			d_o_magnitude_db[i] = 20*log10(d_o_magnitude[i]);
		}

		free(d_o_magnitude);
		free(d_o_magnitude_db);

		// Print out information about the results
		/*
		for (int i=0; i<resultSize; i++) {
			float frequency = (float)i*(float)audio->header.SampleRate/(float)numSamples;
			printf("FFT Result: Frequency: %f: C: %f: I: %f \n"\
					"    Magnitude: %f: Magnitude dB: %f\n", \
					frequency, o_cu_buf[i].x, o_cu_buf[i].y, \
					d_o_magnitude[i], d_o_magnitude_db[i]);
		}*/

		bytesRead = audio->loadData(numSamples);
	}

	// Free all memory
	cufftDestroy(plan);

	cudaFree(d_i_cu_buf);
	cudaFree(d_o_cu_buf);

	cudaFreeHost(o_cu_buf);

}
