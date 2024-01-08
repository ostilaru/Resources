#include <dirent.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gen_pair_dual.h"

// DONE: cmp 2 spec_path
int cmp_spec_path(const SPECNODE *spec1, const SPECNODE *spec2) {
    char *filename1 = strrchr(spec1->filepath, '/');
    char *filename2 = strrchr(spec2->filepath, '/');

    if (filename1 == NULL || filename2 == NULL) {
        fprintf(stderr, "Error: Unable to find filename in filepath.\n");
        return -1;  // You can choose a different way to handle errors
    }

    // Skip the leading '/' character
    filename1++;
    filename2++;

    // Find the first dot in the filenames
    char *dot1 = strchr(filename1, '.');
    char *dot2 = strchr(filename2, '.');

    // Compare the extracted filenames
    return strcmp(dot1, dot2);
}

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

// NOTE: read each file's header and data into *pSpecArray
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

// NOTE: only when head1 == head2 and filenameDate1 = filenameDate2, pair them 
size_t GeneratePair_dual(PAIRNODE *ppairlist, SPECNODE *plist1, size_t cnt1,
                         SPECNODE *plist2, size_t cnt2)
{
  size_t i, j;
  size_t paircnt = 0;

  // tag: for debug
  printf("cnt1: %lu, cnt2: %lu\n", cnt1, cnt2);

  for (i = 0; i < cnt1 && (plist1 + i)->valid == 1; i++)
  {
    for (j = 0; j < cnt2 && (plist2 + j)->valid == 1; j++)
    {
      SEGSPEC *phead1 = &((plist1 + i)->head);
      SEGSPEC *phead2 = &((plist2 + j)->head);

      // TODO: add file name cmp
      if (cmpspec(*phead1, *phead2) == 0 && cmp_spec_path(plist1 + i, plist2 + j) == 0)
      {
        // tag: for debug
        // printf("Debug: i=%lu, j=%lu, plist1->path=%s, plist2->path=%s\n", i, j, (plist1 + i)->filepath, (plist2 + j)->filepath);
        
        ppairlist->srcidx = i;
        ppairlist->staidx = j;
        ppairlist++;
        paircnt++;
      }
    }
  }
  return paircnt;
}
