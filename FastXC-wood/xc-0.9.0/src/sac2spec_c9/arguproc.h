#ifndef _ARGU_PROC_H
#define _ARGU_PROC_H
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <getopt.h>
#include "usage.h"

// typedef struct FreqBand
// {
//     float f1, f2, f3, f4;
//     struct FreqBand *next;
// } FreqBand;

typedef struct ARGUTYPE
{
    char *sacin_lst_1;
    char *sacin_lst_2;
    char *sacin_lst_3;
    char *specout_lst_1;
    char *specout_lst_2;
    char *specout_lst_3;
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

/* Parsing the input argument */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);

#endif