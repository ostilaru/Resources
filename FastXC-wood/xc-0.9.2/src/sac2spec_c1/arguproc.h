#ifndef _ARGU_PROC_H
#define _ARGU_PROC_H

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ARGUTYPE
{
    char *sacin_lst;
    char *specout_lst;
    char *filter_file;
    float seglen;
    float freq_low_limit;
    float freq_high_limit;
    int gpu_id;
    int whitenType;
    int normalizeType;
    int skip_step;
    int thread_num;
} ARGUTYPE;
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);
void usage();

#endif