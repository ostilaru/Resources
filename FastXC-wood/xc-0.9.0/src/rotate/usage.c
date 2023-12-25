#include "usage.h"
/***** display usage *****/
void usage()
{
    fprintf(stderr, "\nUsage:\n");
    fprintf(stderr, "RotateNCF\n");
    fprintf(stderr, "    -I input file list. Each line represents component pair EE EN EZ NE NN NZ ZE ZN ZE oder\n");
    fprintf(stderr, "    -O output file list. Each line represents RR RT RZ RZ TR TT TZ ZR ZT ZZ in order\n");


    fprintf(stderr, "\n\nOur Conventions for NCF\n");
    fprintf(stderr, "    if you cross correlate a with b to get ncf.sac,eg -E -N -Z for a and -e -n -z for station b\n");
    fprintf(stderr, "    then  we take a as source  and positon of a is saved as ncf.evlo and ncf.evla\n");
    fprintf(stderr, "    while we take b as station and positon of b is saved as ncf.stlo and ncf.stla\n");
    fprintf(stderr, "    so:\n");
    fprintf(stderr, "    1. In ncf.sac,the positive part stands for signal traveling from a to b\n");
    fprintf(stderr, "       and the negative part stands for signal traveling from b to a\n");
    fprintf(stderr, "    2. Azimuth is measured clockwise between vector North and vector Source->Station\n");
    fprintf(stderr, "       while Back-Azimuth  clockwise between vector North and vector Station->Source\n");
    fprintf(stderr, "    3. We currently using Net.Sta.Loc to identify one uniq station. for the headers of NCF:\n");
    fprintf(stderr, "       For Virt-Sta in NCF the Net.Sta.Loc and position is\n");
    fprintf(stderr, "            ncf.stla   = b.stla\n");
    fprintf(stderr, "            ncf.stlo   = b.stlo\n");
    fprintf(stderr, "       For Virt-Src in NCF the Net.Sta.Loc and position is\n");
    fprintf(stderr, "            ncf.evla   = a.stla\n");
    fprintf(stderr, "            ncf.evlo   = a.stlo\n");
    fprintf(stderr, "    Delete some usefule but redundant usage to suit the higher level python script calling this C program.\n");
    fprintf(stderr, "      \n");

    fprintf(stderr, "History: update by wangwt@20200223\n");
    fprintf(stderr, "Version: last update by wangjx@20230716\n");
    fprintf(stderr, "\n");
}