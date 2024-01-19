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
  // TODO: add argument for output stack path
  parg->stack_dir = NULL;

  /* check argument */
  if (argc <= 1)
  {
    usage();
    exit(-1);
  }

  /* new stype parsing command line options */
  while ((c = getopt(argc, argv, "A:B:O:S:C:G:X")) != -1)
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
    // TODO: add argument for output stack path
    case 'S':
      parg->stack_dir = optarg;
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

char *createFilePath(const char *stack_dir, const char *sta_pair, const char *base_name) { 
  // find the last dot in base_name
  const char *last_dot = strrchr(base_name, '.');
  
  if (last_dot != NULL) {
    size_t path_len = strlen(stack_dir) + 1 + strlen(sta_pair) + 1 + strlen(base_name) + strlen(".ncf") + 1;
    char *stack_path = (char *)malloc(path_len);

    // Adjusted snprintf to include ".ncf" before the last_dot
    snprintf(stack_path, path_len, "%s/%s/%.*s.ncf%s", stack_dir, sta_pair, (int)(last_dot - base_name), base_name, last_dot);

    return stack_path;
  } else {
    fprintf(stderr, "Error: No dot found in base_name\n");
    return NULL;
  }
}
