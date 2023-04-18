#include "filter_parse.h"

#include "stdio.h"
#include "stdlib.h"

filterParse::filterParse(char* fileName) {
	FILE* fh;
	fh = fopen(fileName, "r");
	if ( fh==NULL ) {
		printf("Error: Filter file is invalid!");
		exit(0);
	}

	// Determine number of values
	double value;
	fseek(fh, 2, SEEK_CUR);

	// Determine length
	this->len = 0;
	int scanSuccess = fscanf(fh, "%lf", &value);
	while (scanSuccess != -1) {
		this->len++;
		fseek(fh, 1, SEEK_CUR);
		scanSuccess = fscanf(fh, "%lf", &value);
	}

	// Read in values
	fseek(fh, 2, SEEK_SET);
	this->coeffArr = (double*) malloc(sizeof(double)*this->len);
	for (int i=0; i<this->len; i++) {
		fscanf(fh, "%lf", &this->coeffArr[i]);
		fseek(fh, 1, SEEK_CUR);
	}
}

filterParse::~filterParse(void) {
	free(this->coeffArr);
}
