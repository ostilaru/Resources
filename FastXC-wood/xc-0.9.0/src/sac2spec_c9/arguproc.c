#include "arguproc.h"

void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument)
{
    if (argc < 2)
    {
        usage();
        exit(1);
    }
    int opt;

    // Set default values for optional arguments
    pargument->whitenType = 0;
    pargument->normalizeType = 0;
    pargument->skip_step = -1;
    pargument->thread_num = 1;

    // Define the option string for getopt

    // Parse command line arguments using getopt
    while ((opt = getopt(argc, argv, "I:O:B:S:L:G:F:W:N:Q:T:")) != -1)
    {
        switch (opt)
        {
        case 'I':
            if (optind + 2 >= argc)
            {
                fprintf(stderr, "Error: Insufficient arguments after -I\n");
                exit(1);
            }
            pargument->sacin_lst_1 = optarg;
            pargument->sacin_lst_2 = argv[optind++];
            pargument->sacin_lst_3 = argv[optind++];
            optind -= 2;
            break;
        case 'O':
            if (optind + 2 >= argc)
            {
                fprintf(stderr, "Error: Insufficient arguments after -O\n");
                exit(1);
            }
            pargument->specout_lst_1 = optarg;
            pargument->specout_lst_2 = argv[optind++];
            pargument->specout_lst_3 = argv[optind++];
            optind -= 2;
            break;
        case 'B':
            pargument->filter_file = optarg;
            break;
        case 'L':
            pargument->seglen = atof(optarg);
            break;
        case 'G':
            pargument->gpu_id = atoi(optarg);
            break;
        case 'F':
        {
            float freq_low_limit, freq_high_limit;
            if (sscanf(optarg, "%f/%f", &freq_low_limit, &freq_high_limit) != 2)
            {
                fprintf(stderr, "Error: Invalid frequency band format\n");
                exit(1);
            }

            // ensure freq_low_limit < freq_high_limit
            if (freq_low_limit >= freq_high_limit)
            {
                fprintf(stderr, "Error: Invalid frequency band range\n");
                exit(1);
            }
            pargument->freq_low_limit = freq_low_limit;
            pargument->freq_high_limit = freq_high_limit;
            break;
        }
        case 'W':
            pargument->whitenType = atoi(optarg);
            break;
        case 'N':
            pargument->normalizeType = atoi(optarg);
            break;
        case 'Q':
            pargument->skip_step = atoi(optarg);
            break;
        case 'T':
            pargument->thread_num = atoi(optarg);
            break;
        default: // '?' or ':' for unrecognized options or missing option arguments
            usage();
            exit(-1);
        }
    }
}