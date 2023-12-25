#include <stdio.h>

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