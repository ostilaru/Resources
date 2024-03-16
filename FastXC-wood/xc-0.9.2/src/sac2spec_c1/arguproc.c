#include "arguproc.h"
#include <stdio.h>

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
    while ((opt = getopt(argc, argv, "I:O:S:L:G:F:W:N:Q:T:B:")) != -1)
    {
        switch (opt)
        {
        case 'I':
            pargument->sacin_lst = optarg;
            break;
        case 'O':
            pargument->specout_lst = optarg;
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

void usage()
{
    fprintf(
        stderr,
        "Usage:\n"
        "sac2spec_c1   -I filelist -O outputlist -L winlength [-G gpu_no] \n"
        "              -W whitenType -N normalizeType -F f1/f2 -B filterfile \n"
        "               [-T num_of_gpu_para] [-Q skip step]\n"
        "Options:\n"
        "   -I  input list of sac files. Every line contains one sac file. The "
        "       input KCMPNM should be set correctly.\n"
        "   -O  output list of segspec files. corresponding to input list '-I' "
        "       option.\n"
        "   -L  length of segment window in seconds. Usually, 7200s (2 hours) is "
        "       used.\n"
        "   -G  Index of Gpu device \n"
        "   -F  FREQ_BANDS Frequency bands for spectral whitenning (in Hz) \n"
        "       using the format f_low_limit/f_high_limit \n"
        "   -W  flag for frequency domain whitening (0, 1, 2, 3)\n"
        "       0: no bandwhiten.\n"
        "       1: whiten before time domain normalization.\n"
        "       2: whiten after time domain normalization.\n"
        "       3: whiten both before and after time domain normalization.\n"
        "   -N  flag for time domain normalization (0, 1, 2)\n"
        "       0: no normalization.\n"
        "       1: runabs normalization.\n"
        "       2: one-bit normalization.\n"
        "   -B  Butterworth filter file. The filter file should be generated \n"
        "   -Q  [optional >=0] will skip certain step. Should be checked carefully\n"
        "   -T  [optional >=0] GPU thread number. Default is 1\n"
        "Last update: wangjx@20240103\n");
}
