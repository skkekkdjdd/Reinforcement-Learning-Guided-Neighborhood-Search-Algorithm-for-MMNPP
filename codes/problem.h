/** data structure for Problem **/
typedef struct {
	int n;							// number of vectors
	int m;							// dimention of vectors
	int k;							// number of sub-sets
	double* weight;					// weights matrix
	double* sum;         			// sums of weights
	double* solution;             	// solution
	double obj_value;				// value of obj function 
} problem_struct;

