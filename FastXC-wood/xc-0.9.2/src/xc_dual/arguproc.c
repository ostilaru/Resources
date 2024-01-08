#include "arguproc.h"

/* parse command line arguments */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  int c;

  parg->src_spectrum_lst = NULL;
  parg->sta_spectrum_lst = NULL;
  parg->ncf_dir = NULL;
  parg->cclength = 0;
  parg->gpu_id = 0;

  /* check argument */
  if (argc <= 1)
  {
    usage();
    exit(-1);
  }

  /* new stype parsing command line options */
  while ((c = getopt(argc, argv, "A:B:O:C:G:X")) != -1)
  {
    switch (c)
    {
    case 'A':
      parg->src_spectrum_lst = optarg;
      break;
    case 'B':
      parg->sta_spectrum_lst = optarg;
      break;
    case 'O':
      parg->ncf_dir = optarg;
      break;
    case 'C':
      parg->cclength = atof(optarg);
      break;
    case 'G':
      parg->gpu_id = atof(optarg);
      break;
    case 'X':
      continue;
    case '?':
    default:
      fprintf(stderr, "Unknown option %c\n", optopt);
      exit(-1);
    }
  }

  /* end of parsing command line arguments */
}
