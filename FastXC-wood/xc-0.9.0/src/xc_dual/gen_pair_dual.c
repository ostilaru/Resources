#include <dirent.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gen_pair_dual.h"

int cmpspec(SEGSPEC hd1, SEGSPEC hd2)
{
  /* first check nseg */
  if (hd1.nstep != hd2.nstep)
  {
    fprintf(stderr, "Different nseg HD1 %d HD2 %d\n", hd1.nstep, hd2.nstep);
    return -1;
  }
  if (hd1.nspec != hd2.nspec)
  {
    fprintf(stderr, "Different nspec HD1 %d HD2 %d\n", hd1.nspec, hd2.nspec);
    return -1;
  }
  if (hd1.dt != hd2.dt)
  {
    fprintf(stderr, "Different dt HD1 %f HD2 %f\n", hd1.dt, hd2.dt);
    return -1;
  }

  return 0;
}

void GenSpecArray(FilePaths *pFileList, SPECNODE *pSpecArray)
{
  SPECNODE *pSpec;

  for (size_t i = 0; i < pFileList->count; i++)
  {
    pSpec = &(pSpecArray[i]);
    strcpy(pSpec->filepath, pFileList->paths[i]);
    if (read_spec_buffer(pSpec->filepath, &(pSpec->head), pSpec->pdata) == NULL)
      continue;

    pSpec->valid = 1;
  }
}

size_t GeneratePair_dual(PAIRNODE *ppairlist, SPECNODE *plist1, size_t cnt1,
                         SPECNODE *plist2, size_t cnt2)
{
  size_t i, j;
  size_t paircnt = 0;
  for (i = 0; i < cnt1 && (plist1 + i)->valid == 1; i++)
  {
    for (j = 0; j < cnt2 && (plist2 + j)->valid == 1; j++)
    {
      SEGSPEC *phead1 = &((plist1 + i)->head);
      SEGSPEC *phead2 = &((plist2 + j)->head);
      if (cmpspec(*phead1, *phead2) == 0)
      {
        ppairlist->srcidx = i;
        ppairlist->staidx = j;
        ppairlist++;
        paircnt++;
      }
    }
  }
  return paircnt;
}
