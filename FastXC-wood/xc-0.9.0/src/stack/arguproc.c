#include "arguproc.h"
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument)
{
    int opt;

    pargument->sac_lst = NULL;
    pargument->out_sac = NULL;
    pargument->normalize = 1;

    struct option long_options[] = {
        {"I", required_argument, NULL, 'I'},
        {"O", required_argument, NULL, 'O'},
        {"A", no_argument, &(pargument->normalize), 0},
        {0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "I:O:A", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'I':
            pargument->sac_lst = strdup(optarg);
            break;
        case 'O':
            pargument->out_sac = optarg;
            break;
        case 'A':
            pargument->normalize = 0;
            break;
        default:
            usage();
            exit(EXIT_FAILURE);
        }
    }
}
