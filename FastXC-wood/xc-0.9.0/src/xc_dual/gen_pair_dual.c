#include <dirent.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gen_pair_dual.h"

int cmp_spec_path_origin(const SPECNODE *spec1, const SPECNODE *spec2) {
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

// DONE: cmp 2 spec_path, extract date part, 2018.001.0000
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

    // Find the last dot in the filenames
    char *lastDot1 = strrchr(filename1, '.');
    char *lastDot2 = strrchr(filename2, '.');

    // between first dot and last dot
    char *nosuffix1 = malloc(lastDot1 - dot1 + 1);
    char *nosuffix2 = malloc(lastDot2 - dot2 + 1);

    if (nosuffix1 == NULL || nosuffix2 == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        free(nosuffix1);
        free(nosuffix2);
        return -1;
    }

    strncpy(nosuffix1, dot1 + 1, lastDot1 - dot1 - 1);
    nosuffix1[lastDot1 - dot1 - 1] = '\0';  // Add null terminator
    strncpy(nosuffix2, dot2 + 1, lastDot2 - dot2 - 1);
    nosuffix2[lastDot2 - dot2 - 1] = '\0';  // Add null terminator

    // Find the second last dot in the filenames
    char *secondLastDot1 = strrchr(nosuffix1, '.');
    char *secondLastDot2 = strrchr(nosuffix2, '.');

    // compare the part between the first and second last dot
    char *date1 = malloc(secondLastDot1 - nosuffix1 + 1);
    char *date2 = malloc(secondLastDot2 - nosuffix2 + 1);

    strncpy(date1, nosuffix1, secondLastDot1 - nosuffix1);
    date1[secondLastDot1 - nosuffix1] = '\0';  // Add null terminator
    strncpy(date2, nosuffix2, secondLastDot2 - nosuffix2);
    date2[secondLastDot2 - nosuffix2] = '\0';  // Add null terminator

    int res = strcmp(date1, date2);

    // free memory
    free(nosuffix1);
    free(nosuffix2);
    free(date1);
    free(date2);

    return res;
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
        ppairlist->srcidx = i;
        ppairlist->staidx = j;
        ppairlist++;
        paircnt++;
      }
    }
  }
  return paircnt;
}
