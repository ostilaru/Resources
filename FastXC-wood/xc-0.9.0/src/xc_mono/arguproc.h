#ifndef __CU_ARG_PROC_H
#define __CU_ARG_PROC_H

#define MAX_GPU_COUNT 100

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "usage.h"

typedef struct ARGUTYPE
{
  char *spectrum_lst; // dir of segspec
  char *ncf_dir;
  float cclength; /* half length of output NCF */
  int gpu_id;     /* GPU ID */
  int xcorr;      /* If do cross-correlation */
} ARGUTYPE;

void usage();
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);

#endif