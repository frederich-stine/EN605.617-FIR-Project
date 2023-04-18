#ifndef __FILTER_PARSE_C__
#define __FILTER_PARSE_C__

using namespace std;

class filterParse {
	public:
		filterParse(char* fileName); 
		~filterParse();	
		int len;
		double* coeffArr;
};

#endif
