#include "gen_pair_mono.h"

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

void GenSpecArray(FilePaths *fileList, SPECNODE *specArray)
{
  SPECNODE *spec;

  for (size_t i = 0; i < fileList->count; i++)
  {
    spec = &(specArray[i]);
    strcpy(spec->filepath, fileList->paths[i]);
    if (read_spec_buffer(spec->filepath, &(spec->head), spec->pdata) == NULL)
      continue;
    spec->valid = 1;
  }
}
typedef struct
{
  char *station;
  int year;
  int julianday;
  int hour;
  int minute;
  char *component;
} FileInfo;

FileInfo extractFileInfo(char *filename)
{
  FileInfo fi;

  fi.station = strtok(filename, ".");
  fi.year = atoi(strtok(NULL, "."));
  fi.julianday = atoi(strtok(NULL, "."));

  char *time = strtok(NULL, ".");
  if (strlen(time) == 4)
  { // Ensure the time format is HHMM
    fi.hour = (time[0] - '0') * 10 + (time[1] - '0');
    fi.minute = (time[2] - '0') * 10 + (time[3] - '0');
  }
  else
  {
    // Handle error: unexpected time format
  }

  fi.component = strtok(NULL, ".");
  // Note: The extension ".segspec" is not stored as it's constant.

  return fi;
}

size_t GeneratePair(PAIRNODE *ppairlist, SPECNODE *plist, size_t spec_cnt, int xcorr)
{
  size_t paircnt = 0;
  for (size_t i = 0; i < spec_cnt && (plist + i)->valid == 1; i++)
  {
    for (size_t j = i; j < spec_cnt && (plist + j)->valid == 1; j++)
    {
      // ensure kstnm1 always be the smaller one
      char *filepath1_copy = strdup((plist + i)->filepath);
      char *filename1 = basename(filepath1_copy);
      FileInfo fi1 = extractFileInfo(filename1);

      char *filepath2_copy = strdup((plist + j)->filepath);
      char *filename2 = basename(filepath2_copy);
      FileInfo fi2 = extractFileInfo(filename2);

      int flag_station = strcmp(fi1.station, fi2.station);

      if (flag_station < 0)
      {
        ppairlist->srcidx = i;
        ppairlist->staidx = j;
      }
      else
      {
        ppairlist->srcidx = j;
        ppairlist->staidx = i;
      }
      free(filepath1_copy);
      free(filepath2_copy);
      ppairlist++;
      paircnt++;
      if (!xcorr) // Only auto-correlation
        break;
    }
  }
  return paircnt;
}
