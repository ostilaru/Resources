#ifndef _RDSPEC_H
#define _RDSPEC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "complex.h"
#include "segspec.h"
#include <dirent.h>
#include <sys/stat.h>  // Include for stat()

// Read in a spec file and store in buffer
complex *read_spec_buffer(char *name, SEGSPEC *hd, complex *buffer);

// Read in a spec file header
int read_spechead(const char *name, SEGSPEC *hd);


// ---------------------------------------- Merge Files ----------------------------------------

// Read all head and all data section Merge Segspec Files:
void readMergedSpecBuffer(const char *filename, complex *spec_buffer, size_t buffer_size);

// Read only one head and all data section Merge Segspec Files:
void readMergedSpecBuffer_onlyOneHead(const char *filename, complex *spec_buffer, size_t buffer_size);

// Read one head of the Merge Segspec Files:
int readMergedSpec_head(const char *filename, SEGSPEC *spec_head);

// Merge all head and all data section into Merge Segspec Files:
int mergeFiles(const char *baseDir, FILE *outputFile);

// Merge only one head and all data section into Merge Segspec Files:
int mergeFiles_onlyOneHead(const char *baseDir, FILE *outputFile, int *firstFile);

#endif
