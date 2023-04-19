#include "wav_parse.h"

#include "stdio.h"
#include "stdlib.h"

wavParse::wavParse(char* fileName) {
	this->fh = fopen(fileName, "r");
	if ( fh==NULL ) {
		printf("Error: Wav file is invalid!");
		exit(0);
	}

	fread(&this->header, 1, sizeof(wavData), fh);
}

size_t wavParse::loadWorkSize(int filterSize) {
	if (this->audioBuf == NULL) {
		free(this->audioBuf);
	}

	this->audioBuf = (double*) malloc(sizeof(double)*this->workSize);
	if (this->audioBuf == NULL) {
		printf("Error: Audio Buf: Malloc failed\n");
		exit(0);
	}

	int16_t* rawAudio = (int16_t*) malloc(sizeof(int16_t)*this->workSize);
	if (this->audioBuf == NULL) {
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

wavParse::~wavParse(void) {
	if (this->audioBuf == NULL) {
		free(this->audioBuf);
	}
	fclose(this->fh);
}

wavWriter::wavWriter(char* fileName) {
	this->fh = fopen(fileName, "w");
	if ( fh==NULL ) {
		printf("Error: Wav file is invalid!");
		exit(0);
	}
}

void wavWriter::writeHeader(wavData* header) {
	fwrite(header, 1, sizeof(wavData), fh);
}

void wavWriter::writeSample(double sample) {
	int16_t rawSample = 0;
	rawSample = (int16_t) sample;
	fwrite(&rawSample, 1, sizeof(int16_t), fh);
}

wavWriter::~wavWriter(void) {
	fclose(this->fh);
}