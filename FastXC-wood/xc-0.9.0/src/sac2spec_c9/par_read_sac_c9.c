#include "par_read_sac_c9.h"

ThreadPoolRead *create_threadpool_read(size_t num_threads)
{
  ThreadPoolRead *pool = malloc(sizeof(ThreadPoolRead));
  pool->threads = malloc(num_threads * sizeof(pthread_t));
  pool->tinfo = malloc(num_threads * sizeof(thread_info_read));
  pool->num_threads = num_threads;
  return pool;
}

void destroy_threadpool_read(ThreadPoolRead *pool)
{
  free(pool->threads);
  free(pool->tinfo);
  free(pool);
}

void *read_in_sacdata_c9(void *arg)
{
  thread_info_read *tinfo = (thread_info_read *)arg;
  for (int i = tinfo->start; i < tinfo->end; i++)
  {
    // reading in sachead
    if (read_sachead(tinfo->InOutList[i].sacpath_1, tinfo->InOutList[i].sac_hd) != 0)
    {
      fprintf(stderr, "Error reading head of SAC file: %s\n",
              tinfo->InOutList[i].sacpath_1);
      return (void *)-1;
    }

    // reading in data 1
    if (read_sac_buffer(tinfo->InOutList[i].sacpath_1, tinfo->InOutList[i].sac_hd,
                        tinfo->InOutList[i].timesignal_1) == NULL)
    {
      fprintf(stderr, "Error reading SAC file: %s\n",
              tinfo->InOutList[i].sacpath_1);
      return (void *)-1;
    }
    // reading in data 2
    if (read_sac_buffer(tinfo->InOutList[i].sacpath_2, tinfo->InOutList[i].sac_hd,
                        tinfo->InOutList[i].timesignal_2) == NULL)
    {
      fprintf(stderr, "Error reading SAC file: %s\n",
              tinfo->InOutList[i].sacpath_2);
      return (void *)-1;
    }
    // reading in data 3
    if (read_sac_buffer(tinfo->InOutList[i].sacpath_3, tinfo->InOutList[i].sac_hd,
                        tinfo->InOutList[i].timesignal_3) == NULL)
    {
      fprintf(stderr, "Error reading SAC file: %s\n",
              tinfo->InOutList[i].sacpath_3);
      return (void *)-1;
    }
  }

  return NULL;
}

int parallel_read_sac_c9(ThreadPoolRead *pool, size_t proccnt, InOutNodeC9 *pInOutList,
                         int num_threads)
{
  printf("[Waiting:] Reading in SAC data, take some time...\n");

  // divide the work
  size_t chunk = proccnt / num_threads;
  size_t remainder = proccnt % num_threads; // calculate the remainder

  size_t start = 0;
  for (size_t i = 0; i < num_threads; i++)
  {
    pool->tinfo[i].start = start;
    // if i<remainder, then add 1 to the chunk size
    pool->tinfo[i].end = start + chunk + (i < remainder ? 1 : 0);
    start = pool->tinfo[i].end; // the start of the next thread is the end of this
                                // current thread
    pool->tinfo[i].InOutList = pInOutList;
    int ret = pthread_create(&pool->threads[i], NULL, read_in_sacdata_c9, &pool->tinfo[i]);

    if (ret)
    {
      fprintf(stderr, "Error creating thread\n");
      return -1;
    }
  }
  for (size_t i = 0; i < pool->num_threads; i++)
  {
    void *status;
    if (pthread_join(pool->threads[i], &status))
    {
      fprintf(stderr, "Error joining thread\n");
      return -1;
    }
    if ((int)(size_t)status == -1)
    {
      fprintf(stderr, "Error occurred in thread while reading SAC file.\n");
      return -1;
    }
  }

  printf("Done reading in SAC data.\n");
  return 0;
}