/** Frederich Stine - EN605.617 FIR Project
 *  wav_parse.h
 *  
 *  This is the header file for the wavParse and wavWrite classes
 * 
 *  These classes handle reading in wav files for FIR filtering and 
 *  FFT calculations.
 *  These classes also handle writing the filtered data to a new file.
 */ 

#ifndef __WAV_PARSE_C__
#define __WAV_PARSE_C__

#include "stdint.h"
#include "stdio.h"

/******************* Data Type Definitions ********************/
// Structure representing the header of a wav file
typedef struct {
	uint8_t  ChunkID[4];
	uint32_t ChunkSize;
	uint8_t  Format[4];
	uint8_t  SubChunkID[4];
	uint32_t SubChunkSize;
	uint16_t AudioFormat;
	uint16_t NumChannels;
	uint32_t SampleRate;
	uint32_t ByteRate;
	uint16_t BlockAlign;
	uint16_t BitsPerSample;
	uint8_t  SubChunk2ID[4];
	uint32_t SubChunk2Size;
} wavData;

/** Wav Parse class definitions
*/
class wavParse {
	public:
		// Wav parse constructor takes a file name
		wavParse(char* fileName); 
		// Wav parse deconstructor
		~wavParse();
		
		// Function for loading data for FIR filtering
		size_t loadWorkSize(int filterSize);
		// Function for loading dat for FFT calculation
		size_t loadData(int count);
		// Function to reset the file pointer
		void reset();
		// Function to print data about the file
		void printInfo();

		// Header of wav file
		wavData header;
		// Filename of wav file
		char* fileName = NULL;
		// Buffer for loading of audio data
		double* audioBuf = NULL;
		// Work size of FIR filter
		int workSize = 10000;
	private:
		// File handle
		FILE* fh;
		// Bool to pad data for FFT
		bool firstLoad = true;
};

/** Wav Writer class definitions
*/
class wavWrite {
	public:
		// Wav write constructor takes a file name
		wavWrite(char* fileName); 
		// Wav write deconstructor
		~wavWrite();
		
		// Funtcion to write the file header
		void writeHeader(wavData* header);
		// Funtion to write a single sample
		void writeSample(double sample);
		// Function to write many samples
		void writeBulk(double* sample, int sampleCount);
	private:
		// File handle
		FILE* fh;
};


#endif
