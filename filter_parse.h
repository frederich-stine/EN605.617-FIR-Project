/** Frederich Stine - EN605.617 FIR Project
 *  filter_parse.h
 *  
 *  This is the header file for the filterParse class
 * 
 *  This class reads in filters exported by pyfda.
 */ 

#ifndef __FILTER_PARSE_C__
#define __FILTER_PARSE_C__

using namespace std;

/** Filter Parse class definition
*/
class filterParse {
	public:
		// Constructor takes in a file name
		filterParse(char* fileName);
		// Deconstructor
		~filterParse();
		// Length of filter
		int len;
		// Array of filter elements
		double* coeffArr;
};

#endif
