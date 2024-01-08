#ifndef _IN_OUT_NODE_c9_H
#define _IN_OUT_NODE_c9_H
#include "sac.h"
#include <stdio.h>
#include "complex.h"
#include "segspec.h"

typedef struct InOutNodeC9
{
    char *sacpath_1;
    char *sacpath_2;
    char *sacpath_3;

    char *specpath_1;
    char *specpath_2;
    char *specpath_3;

    float *timesignal_1;
    float *timesignal_2;
    float *timesignal_3;

    SACHEAD *sac_hd;
    SEGSPEC *segspec_hd;

    complex *spectrum_1;
    complex *spectrum_2;
    complex *spectrum_3;

    float nstep;
    float nspec;
    float df;
    float dt;
} InOutNodeC9;

#endif