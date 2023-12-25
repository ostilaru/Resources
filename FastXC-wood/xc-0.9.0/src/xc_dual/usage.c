#include "usage.h"
void usage()
{
    fprintf(
        stderr,
        "\nUsage:\n"
        "specxc_mg -A virt_src_lst -B virt_sta_dir -C halfCCLength -O "
        "outputdir -G gpu num\n"
        "Options:\n"
        "    -A Specify the list file of input files for the 1st station, eg virtual "
        "source\n"
        "    -B Specify the list file of input files for the 2nd station, eg virtual "
        "station\n"
        "    -C Half of cclenth (in seconds).\n"
        "    -O Output directory for NCF files as sac format\n"
        "    -G ID of Gpu device to be launched \n"
        "Version:\n"
        "  last update by wangjx@20230504\n"
        "  cuda version\n");
}