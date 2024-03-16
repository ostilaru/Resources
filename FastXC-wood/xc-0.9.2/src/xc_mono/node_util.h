#ifndef _NODE_UTIL_H
#define _NODE_UTIL_H
#include "config.h"
#include "segspec.h"
#include "sac.h"
#include "complex.h"
#include <stdio.h>

typedef struct FilePaths
{
    char **paths;
    int count;
} FilePaths;

typedef struct SPECNODE
{
    int valid;
    char filepath[MAXLINE];
    SEGSPEC head;
    complex *pdata;
} SPECNODE;

typedef struct PAIRNODE
{
    size_t srcidx;
    size_t staidx;
    SACHEAD headncf;
} PAIRNODE;

#endif