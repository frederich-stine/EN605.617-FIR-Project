/** Frederich Stine - EN605.617 FIR Project
 *  wav_parse.h
 *  
 *  This is the implementation file for the wavParse and wavWrite classes
 * 
 *  These classes handle reading in wav files for FIR filtering and 
 *  FFT calculations.
 *  These classes also handle writing the filtered data to a new file.
 */ 

#include "wav_parse.h"

#include "stdio.h"
#include "stdlib.h"

/******************* Class Function Definitions ********************/
/** Wav Parse constructor 
 * 
 * This constructor takes in a file name and reads in the header
*/
wavParse::wavParse(char* fileName) {
	this->fh = fopen(fileName, "rb");
	if ( this->fh==NULL ) {
		printf("Error: Wav file is invalid!");
		exit(0);
	}

	fread(&this->header, 1, sizeof(wavData), this->fh);
}

/** Wav Parse load work size
 * 
 *  This function loads the correct amount of data to perform a FIR convolution
*/
size_t wavParse::loadWorkSize(int filterSize) {
	// Free buffer before reloading data
	if (this->audioBuf != NULL) {
		free(this->audioBuf);
		this->audioBuf = NULL;
	}

	// Reallocate buffer to correct size
	this->audioBuf = (double*) malloc(sizeof(double)*this->workSize);
	if (this->audioBuf == NULL) {
		printf("Error: Audio Buf: Malloc failed\n");
		exit(0);
	}

	// Allocate raw audio buffer
	int16_t* rawAudio = (int16_t*) malloc(sizeof(int16_t)*this->workSize);
	if (rawAudio == NULL) {
		printf("Error: Raw Audio Buf: Malloc failed\n");
		exit(0);
	}

	// If the first load is true we need to pad the buffer
	size_t bytesRead = 0;
	if (this->firstLoad == true) {
		// Pad buffer
		for (int i=0; i<filterSize; i++) {
			this->audioBuf[i] = 0.0;
		}
		
		// Read in remaining data
		bytesRead = fread(&rawAudio[filterSize], 1, 
			this->workSize*sizeof(int16_t) - filterSize*sizeof(int16_t), this->fh);

		// Convert to double
		for (int i=filterSize; i<this->workSize; i++) {
			this->audioBuf[i] = (double) rawAudio[i];
		}

		// Free rawaudio, set firstLoad, and return bytes read
		this->firstLoad = false;
		free(rawAudio);
		return (bytesRead/(sizeof(int16_t))+filterSize);
	}

	// Seek backwards to the first data point to process as the FIR algorithm
	// does not compute the values for the last filter size
	fseek(fh, filterSize*-1*sizeof(int16_t), SEEK_CUR);

	// Read in values to the buffer
	bytesRead = fread(rawAudio, 1, this->workSize*sizeof(int16_t), this->fh);

	// Convert to double
	for (int i=0; i<this->workSize; i++) {
		this->audioBuf[i] = (double) rawAudio[i];
	}

	// Free raw audio and return quantity read
	free(rawAudio);
	return bytesRead/(sizeof(int16_t));
}

/** Wav Parse load data
 * 
 *  This function data to perform a FFT
*/
size_t wavParse::loadData(int count) {
	// Free buffer before reloading data
	if (this->audioBuf != NULL) {
		free(this->audioBuf);
		this->audioBuf = NULL;
	}

	// Reallocate buffer to correct size
	this->audioBuf = (double*) malloc(sizeof(double)*count);
	if (this->audioBuf == NULL) {
		printf("Error: Audio Buf: Malloc failed\n");
		exit(0);
	}

	// Allocate raw audio buffer
	int16_t* rawAudio = (int16_t*) malloc(sizeof(int16_t)*count);
	if (rawAudio == NULL) {
		printf("Error: Raw Audio Buf: Malloc failed\n");
		exit(0);
	}

	// Read in specified size of data
	size_t bytesRead = 0;
	bytesRead = fread(rawAudio, 1, count*sizeof(int16_t), this->fh);

	// Convert to double
	for (int i=0; i<count; i++) {
		this->audioBuf[i] = (double) rawAudio[i];
	}

	// Free raw audio and return quantity read
	free(rawAudio);
	return bytesRead/(sizeof(int16_t));
}

/** Wav Parse reset function
 * 
 *  This function resets the file pointer to the first audio sample
*/
void wavParse::reset() {
	// Reset file pointer and firstLoad
	fseek(this->fh, sizeof(wavData), SEEK_SET);
	this->firstLoad = true;
}

/** Wav Parse print information function 
 *  
 *  This funtion prints some data about the wav file
*/
void wavParse::printInfo() {
	printf("----- Wav Info -----\n");
	printf("ChunkSize: %d\n", this->header.ChunkSize);
	printf("SubChunkSize: %d\n", this->header.SubChunkSize);
	printf("AudioFormat: %d\n", this->header.AudioFormat);
	printf("NumChannels: %d\n", this->header.NumChannels);
	printf("SampleRate: %d\n", this->header.SampleRate);
	printf("ByteRate: %d\n", this->header.ByteRate);
	printf("BlockAlign: %d\n", this->header.BlockAlign);
	printf("BitsPerSample: %d\n", this->header.BitsPerSample);
	printf("SubChunk2Size %d\n\n", this->header.SubChunkSize);
}

/** Wav Parse deconstructor 
 *  
 *  This deconstructor closes the file, and frees the audio buffer
*/
wavParse::~wavParse(void) {
	// Free audio buffer
	if (this->audioBuf != NULL) {
		free(this->audioBuf);
	}
	// Close file
	fclose(this->fh);
}

/** Wav Write constructor 
 * 
 *  This constructor takes in a file name
*/
wavWrite::wavWrite(char* fileName) {
	// Open file for reading
	this->fh = fopen(fileName, "wb");
	if ( this->fh==NULL ) {
		printf("Error: Wav file is invalid!\n");
		exit(0);
	}
}

/** Wav Write write header 
 * 
 *  This function writes the wav header specified to the file pointer
*/
void wavWrite::writeHeader(wavData* header) {
	// Write header
	fwrite(header, 1, sizeof(wavData), fh);
}

/** Wav Write write sample
 * 
 *  This function writes a single sample to the file pointer
*/
void wavWrite::writeSample(double sample) {
	// Convert sample to integer and write to the file
	int16_t rawSample = 0;
	rawSample = (int16_t) sample;
	fwrite(&rawSample, 1, sizeof(int16_t), this->fh);
}

/** Wav Write write bulk
 * 
 *  This function writes a bulk quantity of samples to the file pointer
*/
void wavWrite::writeBulk(double* samples, int sampleCount) {
	// Allocate buffer for integer samples
	int16_t* rawSamples = (int16_t*) malloc(sampleCount*sizeof(int16_t));

	// Convert all samples to integers
	for (int i=0; i<sampleCount; i++) {
		rawSamples[i] = (int16_t) samples[i];
	}

	// Write and free buffer
	fwrite(rawSamples, 1, sampleCount*sizeof(int16_t), this->fh);
	free (rawSamples);
}


/** Wav Write deconstructor
 * 
 *  This function closes the file
*/
wavWrite::~wavWrite(void) {
	// Close file
	fclose(this->fh);
}