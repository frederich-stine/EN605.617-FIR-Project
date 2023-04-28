/** Frederich Stine - EN605.617 FIR Project
 *  filter_parse.cpp
 *  
 *  This is the implementation file for the filterParse class
 * 
 *  This class reads in filters exported by pyfda.
 */ 

#include "filter_parse.h"

#include "stdio.h"
#include "stdlib.h"

/******************* Class Function Definitions ********************/
/** Filter Parse constructor 
 * 
 *  This constructor takes in a file name and parses the contents to
 *  a buffer for use with FIR convolutions. 
*/
filterParse::filterParse(char* fileName) {
	// Open file
	FILE* fh;
	fh = fopen(fileName, "r");
	if ( fh==NULL ) {
		printf("Error: Filter file is invalid!\n");
		exit(0);
	}

	// Determine number of values
	double value;
	fseek(fh, 2, SEEK_CUR);

	// Determine length
	this->len = 0;
	int scanSuccess = fscanf(fh, "%lf", &value);
	while (scanSuccess != -1) {
		// Add length of values
		this->len++;
		fseek(fh, 1, SEEK_CUR);
		scanSuccess = fscanf(fh, "%lf", &value);
	}

	// Read in values
	fseek(fh, 2, SEEK_SET);
	this->coeffArr = (double*) malloc(sizeof(double)*this->len);
	for (int i=0; i<this->len; i++) {
		// Read values into the array
		fscanf(fh, "%lf", &this->coeffArr[i]);
		fseek(fh, 1, SEEK_CUR);
	}
}

/** Filter Parse deconstructor
 * 
 *  This deconstructor frees the coefficient buffer
*/
filterParse::~filterParse(void) {
	// Free coefficient buffer
	free(this->coeffArr);
}
