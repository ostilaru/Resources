#ifndef _CREATE_DIR_H
#define _CREATE_DIR_H

// Recursively creates directories
#include <libgen.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int create_parent_dir(const char *path);

#endif