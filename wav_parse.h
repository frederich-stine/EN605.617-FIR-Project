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

class wavParse {
	public:
		wavParse(char* fileName); 
		~wavParse();
		
		size_t loadWorkSize(int filterSize);
		size_t loadData(int count);
		void reset();

		wavData header;
		char* fileName = NULL;
		double* audioBuf = NULL;
		int workSize = 10000;
		FILE* fh;
	private:
		bool firstLoad = true;
};

class wavWrite {
	public:
		wavWrite(char* fileName); 
		~wavWrite();
		
		void writeHeader(wavData* header);
		void writeSample(double sample);
		void writeBulk(double* sample, int sampleCount);
		FILE* fh;

};


#endif
