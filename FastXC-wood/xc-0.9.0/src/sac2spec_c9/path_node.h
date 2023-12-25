#ifndef PATH_NODE_H
#define PATH_NODE_H
#include <stdio.h>
#include "config.h"

typedef struct PathNode
{
    char *path;
    struct PathNode *next;
} PathNode;

typedef struct FilePathArray
{
    char **paths;
    int count;
} FilePathArray;

#endif