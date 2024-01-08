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

/* display usage */
void usage()
{
	fprintf(stderr, "Usage:\n"
					" sac2spec_c9  -IE saclistE -IN filelistN -IZ filelistZ \n"
					"               -OE speclistE -ON speclistN -OZ speclistZ \n"
					"                -L winlength -G gpu_no -W whiten_type -N normalize type\n"
					"                -F freq_bands -Q skip_step\n"

					" Options:\n"
					"   -I*  input list of sac files of [Z,N,E] component.\n"
					"		 Lines having the same line number in different filelist should represent daily SAC file\n"
					"		 of different components with same staion name and date.\n"
					"    	 Also, the input KCMPNM should be set right.The input files must have the same\n"
					"     	 byte-order, little-endian is suggested, and the same NPTS\n"
					"   -O*  output spec list of segment spectrum files. Corresponding to -I option\n"
					"   -L   length of segment window in seconds. Usually I use 7200s 2 hour\n"
					"   -G   Index of Gpu device \n"
					"   -F   FREQ_BANDS Frequency bands for spectral whitenning (in Hz) \n"
					"        using the format f_low_limit/f_high_limit \n"
					"   -W   Whiten type.\n"
					"		   0: donot do any whitenning or normalizaion\n"
					"		   1: do bandwhitening before time domain normalization\n"
					"		   2: do bandwhitening after time domain normalization\n"
					"         Set as 3: do bandwhitenning before and after time domain normalization\n"
					"   -N    Normalization type\n"
					"          0: no normalization.\n"
					"          1: runabs normalization.\n"
					"          2: one-bit normalization.\n"
					"   -B  Butterworth filter file. The filter file should be generated \n"
					"   -Q   skip segment step. Default is 1, which means we do not skip any step\n"
					"   -T  [optional >=0] GPU thread number. Default is 1\n"
					"   Last updated by wangjx@20231121\n");
}