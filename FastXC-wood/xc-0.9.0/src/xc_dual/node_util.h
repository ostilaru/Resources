#ifndef NODE_UTIL_H
#define NODE_UTIL_H

#include "config.h"
#include "segspec.h"
#include "sac.h"
#include "complex.h"
#include <stdio.h>

typedef struct FilePaths
{
    // NOTE: **paths is 2D array, each row is a path
    char **paths;
    int count;
} FilePaths;

// NOTE: STANODE is used to store the station information
typedef struct SPECNODE
{
    int valid;
    char filepath[MAXLINE];
    SEGSPEC head;
    complex *pdata;
} SPECNODE;

// NOTE: PAIRNODE is used to store the pair of SAC files
typedef struct PAIRNODE
{
    size_t srcidx;
    size_t staidx;
    SACHEAD headncf;
} PAIRNODE;

#endif