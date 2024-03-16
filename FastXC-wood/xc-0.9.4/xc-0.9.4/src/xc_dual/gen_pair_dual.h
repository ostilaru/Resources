#ifndef _GEN_PAIR_DUAL_H
#define _GEN_PAIR_DUAL_H

#include <limits.h>
#include <stdio.h>
#include "read_segspec.h"
#include "node_util.h"
#include "complex.h"
#include "sac.h"
#include "segspec.h"

// compare two SEGSPEC
int cmpspec(SEGSPEC hd1, SEGSPEC hd2);

// read in spec files and stor in SPECNODE
void GenSpecArray(FilePaths *pFileList, SPECNODE *pSpecArray);

// check if two stations are paired
size_t GeneratePair_dual(PAIRNODE *ppairlist, SPECNODE *plist1, size_t cnt1,
                         SPECNODE *plist2, size_t cnt2);

#endif
