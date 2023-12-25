#ifndef _ARGU_PROC_H
#define _ARGU_PROC_H

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// typedef struct FreqBand
// {
//     float f1, f2, f3, f4;
//     struct FreqBand *next;
// } FreqBand;

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

void usage();

/* Parsing the input argument */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);

/* Find the optimal transform length for the input length */
int findOptimalTransformLength(int inputLength);

/* Find the largest optimal transform length for the input length */
int findLargestTransformLengthBelow(int inputLength, int *error);

/* Compute the flag for whiten and normalization */
void checkwhiten(int whitenType, int normalizeType, int *wh_before,
                 int *wh_after, int *do_runabs, int *do_onebit);

#endif