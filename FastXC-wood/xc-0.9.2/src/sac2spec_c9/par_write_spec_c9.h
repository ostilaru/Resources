#ifndef _PARALLEL_WRITE_SPEC_C9_H_
#define _PARALLEL_WRITE_SPEC_C9_H_

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "in_out_node_c9.h"

typedef struct thread_info_write
{
    size_t start;
    size_t end;
    InOutNodeC9 *InOutList;
} thread_info_write;

typedef struct ThreadPoolWrite
{
    pthread_t *threads;
    thread_info_write *tinfo;
    size_t num_threads;
} ThreadPoolWrite;

ThreadPoolWrite *create_threadpool_write(size_t num_threads);

void destroy_threadpool_write(ThreadPoolWrite *pool);

int parallel_write_spec_c9(ThreadPoolWrite *pool, size_t h_proccnt, InOutNodeC9 *pInOutList, int num_threads);

#endif