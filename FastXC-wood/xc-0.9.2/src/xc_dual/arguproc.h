#ifndef __CU_ARG_PROC_H
#define __CU_ARG_PROC_H

#define MAX_GPU_COUNT 100

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct ARGUTYPE
{
  /* input list file of -ALST and -BLST */
  char *src_spectrum_lst;
  char *sta_spectrum_lst;
  /* output dir for CC vector */
  char *ncf_dir;
  float cclength; /* half length of output NCF */
  size_t gpu_id;     /* GPU ID */
} ARGUTYPE;

void usage();
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);

#endif