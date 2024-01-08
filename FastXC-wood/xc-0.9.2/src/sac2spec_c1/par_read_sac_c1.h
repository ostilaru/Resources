#ifndef _PARALLEL_READ_SAC_C1_H_
#define _PARALLEL_READ_SAC_C1_H_
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "path_node.h"
#include "in_out_node_c1.h"
#include "sac.h"

typedef struct thread_info_read
{
  int start;
  int end;
  InOutNodeC1 *InOutList;
} thread_info_read;

typedef struct ThreadPoolRead
{
  pthread_t *threads;
  thread_info_read *tinfo;
  size_t num_threads;
} ThreadPoolRead;

ThreadPoolRead *create_threadpool_read(size_t num_threads);

int parallel_read_sac_c1(ThreadPoolRead *pool, size_t proccnt, InOutNodeC1 *pInOutList,
                         int num_threads);

void destroy_threadpool_read(ThreadPoolRead *pool);

#endif
