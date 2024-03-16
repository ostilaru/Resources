#include "arguproc.h"

/* parse command line arguments */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  int c;

  parg->spectrum_lst = NULL;
  parg->ncf_dir = NULL;
  parg->cclength = 0;
  parg->gpu_id = 0;
  parg->xcorr = 0;

  /* check argument */
  if (argc <= 1)
  {
    usage();
    exit(-1);
  }

  /* new stype parsing command line options */
  while ((c = getopt(argc, argv, "A:O:C:XG:")) != -1)
  {
    switch (c)
    {
    case 'A':
      parg->spectrum_lst = optarg;
      break;
    case 'O':
      parg->ncf_dir = optarg;
      break;
    case 'C':
      parg->cclength = atof(optarg);
      break;
    case 'X':
      parg->xcorr = 1;
      break;
    case 'G':
      parg->gpu_id = atof(optarg);
      break;
    case '?':
    default:
      fprintf(stderr, "Unknown option %c\n", optopt);
      exit(-1);
    }
  }

  /* end of parsing command line arguments */
}
