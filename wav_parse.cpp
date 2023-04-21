#include "wav_parse.h"

#include "stdio.h"
#include "stdlib.h"

wavParse::wavParse(char* fileName) {
	this->fh = fopen(fileName, "rb");
	if ( this->fh==NULL ) {
		printf("Error: Wav file is invalid!");
		exit(0);
	}

	fread(&this->header, 1, sizeof(wavData), this->fh);
}

size_t wavParse::loadWorkSize(int filterSize) {
	if (this->audioBuf != NULL) {
		free(this->audioBuf);
		this->audioBuf = NULL;
	}

	this->audioBuf = (double*) malloc(sizeof(double)*this->workSize);
	if (this->audioBuf == NULL) {
		printf("Error: Audio Buf: Malloc failed\n");
		exit(0);
	}

	int16_t* rawAudio = (int16_t*) malloc(sizeof(int16_t)*this->workSize);
	if (rawAudio == NULL) {
		printf("Error: Raw Audio Buf: Malloc failed\n");
		exit(0);
	}

	size_t bytesRead = 0;
	if (this->firstLoad == true) {
		for (int i=0; i<filterSize; i++) {
			this->audioBuf[i] = 0.0;
		}
		
		bytesRead = fread(&rawAudio[filterSize], 1, 
			this->workSize*sizeof(int16_t) - filterSize*sizeof(int16_t), this->fh);

		for (int i=filterSize; i<this->workSize; i++) {
			this->audioBuf[i] = (double) rawAudio[i];
		}

		this->firstLoad = false;
		free(rawAudio);
		return (bytesRead/(sizeof(int16_t))+filterSize);
	}

	fseek(fh, filterSize*-1*sizeof(int16_t), SEEK_CUR);
	bytesRead = fread(rawAudio, 1, this->workSize*sizeof(int16_t), this->fh);

	for (int i=0; i<this->workSize; i++) {
		this->audioBuf[i] = (double) rawAudio[i];
	}

	free(rawAudio);
	return bytesRead/(sizeof(int16_t));
}

size_t wavParse::loadData(int count) {
	if (this->audioBuf != NULL) {
		free(this->audioBuf);
		this->audioBuf = NULL;
	}

	this->audioBuf = (double*) malloc(sizeof(double)*count);
	if (this->audioBuf == NULL) {
		printf("Error: Audio Buf: Malloc failed\n");
		exit(0);
	}

	int16_t* rawAudio = (int16_t*) malloc(sizeof(int16_t)*count);
	if (rawAudio == NULL) {
		printf("Error: Raw Audio Buf: Malloc failed\n");
		exit(0);
	}

	size_t bytesRead = 0;
	bytesRead = fread(rawAudio, 1, count*sizeof(int16_t), this->fh);

	for (int i=0; i<count; i++) {
		this->audioBuf[i] = (double) rawAudio[i];
	}

	free(rawAudio);
	return bytesRead/(sizeof(int16_t));
}

void wavParse::reset() {
	fseek(this->fh, sizeof(wavData), SEEK_SET);
	this->firstLoad = true;
}

long int wavParse::tell() {
	return ftell(this->fh);
}

void wavParse::flush() {
	if (this->fh == NULL) {
		return;
	}

	fflush(this->fh);
	fclose(this->fh);

	this->fh = fopen(this->fileName, "rwb");
	if ( this->fh==NULL ) {
		printf("Error: Wav file is invalid!");
		exit(0);
	}
}

wavParse::~wavParse(void) {
	if (this->audioBuf != NULL) {
		free(this->audioBuf);
	}
	fclose(this->fh);
}

wavWrite::wavWrite(char* fileName) {
	this->fh = fopen(fileName, "wb");
	if ( this->fh==NULL ) {
		printf("Error: Wav file is invalid!");
		exit(0);
	}
}

void wavWrite::writeHeader(wavData* header) {
	fwrite(header, 1, sizeof(wavData), fh);
}

void wavWrite::writeSample(double sample) {
	int16_t rawSample = 0;
	rawSample = (int16_t) sample;
	fwrite(&rawSample, 1, sizeof(int16_t), this->fh);
}

void wavWrite::writeBulk(double* samples, int sampleCount) {
	int16_t* rawSamples = (int16_t*) malloc(sampleCount*sizeof(int16_t));

	for (int i=0; i<sampleCount; i++) {
		rawSamples[i] = (int16_t) samples[i];
	}

	fwrite(rawSamples, 1, sampleCount*sizeof(int16_t), this->fh);
	free (rawSamples);
}

wavWrite::~wavWrite(void) {
	fclose(this->fh);
}