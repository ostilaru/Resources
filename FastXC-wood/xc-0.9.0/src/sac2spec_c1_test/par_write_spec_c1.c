#include "par_write_spec_c1.h"

ThreadPoolWrite *create_threadpool_write(size_t num_threads)
{
  ThreadPoolWrite *pool = malloc(sizeof(ThreadPoolWrite));
  pool->threads = malloc(num_threads * sizeof(pthread_t));
  pool->tinfo = malloc(num_threads * sizeof(thread_info_write));
  pool->num_threads = num_threads;
  return pool;
}

void destroy_threadpool_write(ThreadPoolWrite *pool)
{
  free(pool->threads);
  free(pool->tinfo);
  free(pool);
}

void *write_out_spec_c1(void *arg)
{
  thread_info_write *tinfo = (thread_info_write *)arg;
  for (int i = tinfo->start; i < tinfo->end; i++)
  {
    // print specpatg
    FILE *fid = fopen(tinfo->InOutList[i].specpath, "wb");
    if (fid == NULL)
    {
      perror("Error opening file!\n");
      continue;
    }

    SEGSPEC *segspec_hd = tinfo->InOutList[i].segspec_hd;
    complex *specdata = tinfo->InOutList[i].spectrum;
    segspec_hd->stla = (tinfo->InOutList[i].sac_hd)->stla;
    segspec_hd->stlo = (tinfo->InOutList[i].sac_hd)->stlo;
    segspec_hd->nstep = tinfo->InOutList[i].nstep;
    segspec_hd->nspec = tinfo->InOutList[i].nspec;
    segspec_hd->df = tinfo->InOutList[i].df;
    segspec_hd->dt = tinfo->InOutList[i].dt;
    int nspec = segspec_hd->nspec;
    int nstep = segspec_hd->nstep;
    int write_size = sizeof(complex) * nspec * nstep;

    fwrite(segspec_hd, sizeof(SEGSPEC), 1, fid);

    size_t written = fwrite(specdata, write_size, 1, fid);
    if (written != 1)
    {
      perror("Error writing file!\n");
    }

    // fwrite(specdata, write_size, 1, fid);
    fclose(fid);
  }

  return NULL;
}

int parallel_write_spec_c1(ThreadPoolWrite *pool, size_t h_proccnt,
                           InOutNodeC1 *pInOutList, int num_threads)
{
  // divide the work
  size_t chunk = h_proccnt / num_threads;
  size_t remainder = h_proccnt % num_threads; // calculate the remainder
  printf("Writing spec data, waiting...");
  size_t start = 0;
  for (size_t i = 0; i < num_threads; i++)
  {
    pool->tinfo[i].start = start;
    // if i<remainder, then add 1 to the chunk size
    pool->tinfo[i].end = start + chunk + (i < remainder ? 1 : 0);
    start = pool->tinfo[i].end; // the start of the next thread is the end of this current thread

    pool->tinfo[i].InOutList = pInOutList;

    int ret = pthread_create(&pool->threads[i], NULL, write_out_spec_c1, &pool->tinfo[i]);
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
  }

  printf("Finish writing the output %lu spectra\n", h_proccnt);
  return 0;
}
