#include "usage.h"
/* display usage */
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
        "Last update: wangjx@20231121\n");
}
/* end of display usage */