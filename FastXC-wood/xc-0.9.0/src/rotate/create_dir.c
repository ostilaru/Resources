#include "create_dir.h"

int create_parent_dir(const char *path)
{
    char *path_copy = strdup(path);
    char *parent_dir = dirname(path_copy);

    if (access(parent_dir, F_OK) == -1)
    {
        create_parent_dir(parent_dir);
        if (mkdir(parent_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST)
        {
            free(path_copy);
            return -1;
        }
    }

    free(path_copy);
    return 0;
}
