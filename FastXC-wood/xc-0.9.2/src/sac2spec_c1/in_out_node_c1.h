#ifndef _IN_OUT_NODE_H
#define _IN_OUT_NODE_H

#include "sac.h"
#include "segspec.h"
#include "complex.h"
#include "config.h"

typedef struct InOutNodeC1
{
    char *sacpath;
    char *specpath;
    float *timesignal;
    SACHEAD *sac_hd;
    SEGSPEC *segspec_hd;
    complex *spectrum;
    float nstep;
    float nspec;
    float df;
    float dt;
} InOutNodeC1;

#endif