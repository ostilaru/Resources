#ifndef _GEN_PAIR_MONO_H
#define _GEN_PAIR_MONO_H

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>
#include "node_util.h"
#include "sac.h"
#include "segspec.h"
#include "read_segspec.h"

// compare two SEGSPEC
int cmpspec(SEGSPEC hd1, SEGSPEC hd2);

// generate a list of station pairs and set the data pointer
void GenSpecArray(FilePaths *fileList, SPECNODE *specArray);

// check if two stations are paired
size_t GeneratePair(PAIRNODE *ppairlist, SPECNODE *plist, size_t spec_cnt, int xcorr);

#endif
