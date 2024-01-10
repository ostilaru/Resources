#include "usage.h"
void usage()
{
    fprintf(
        stderr,
        "\nUsage:\n"
        "specxc_mg -A spec_lst -O out_dir -C halfCCLength -G gpu_id [-X do cross-correlation] \n"
        "Options:\n"
        "    -A Input spectrum list\n"
        "    -O Output directory for NCF files as sac format\n"
        "    -C Half of cclenth (in seconds).\n"
        "    -G ID of Gpu device to be launched \n"
        "    -X Optional. If set, do cross-correlation; else, only do "
        "auto-correlation.\n"
        "Version:\n"
        "  last update by wangjx@20230627\n"
        "  cuda version\n");
}