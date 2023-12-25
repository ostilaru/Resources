#include "find_whiten_flag.h"

void find_whiten_flag(int whitenType, int normalizeType, int *wh_before,
                      int *wh_after, int *do_runabs, int *do_onebit)
{
    // Determine bandwhiten flags
    switch (whitenType)
    {
    case 0:
        *wh_before = 0;
        *wh_after = 0;
        break;
    case 1:
        *wh_before = 1;
        *wh_after = 0;
        break;
    case 2:
        *wh_before = 0;
        *wh_after = 1;
        break;
    case 3:
        *wh_before = 1;
        *wh_after = 1;
        break;
    default:
        // Invalid value for bandwhiten
        printf("Invalid value for bandwhiten\n");
        // Handle error or set default values
        break;
    }

    // Determine normalization flags
    switch (normalizeType)
    {
    case 0:
        *do_runabs = 0;
        *do_onebit = 0;
        break;
    case 1:
        *do_runabs = 1;
        *do_onebit = 0;
        break;
    case 2:
        *do_runabs = 0;
        *do_onebit = 1;
        break;
    default:
        // Invalid value for normalization
        printf("Invalid value for normalization\n");
        // Handle error or set default values
        break;
    }
}
