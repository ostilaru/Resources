#ifndef _CU_ONEBIT_H_
#define _CU_ONEBIT_H_
#include "cuda.util.cuh"

void onebit(float *d_data, int nseg, int proccnt);

#endif