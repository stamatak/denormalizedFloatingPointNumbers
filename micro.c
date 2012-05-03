#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <xmmintrin.h>
#include <stdarg.h>
#include <pmmintrin.h>
#include <assert.h>
#include <stdint.h>
#include <emmintrin.h>

static double gettime(void)
{
#ifdef WIN32
  time_t tp;
  struct tm localtm;
  tp = time(NULL);
  localtm = *localtime(&tp);
  return 60.0*localtm.tm_min + localtm.tm_sec;
#else
  struct timeval ttime;
  gettimeofday(&ttime , NULL);
  return ttime.tv_sec + ttime.tv_usec * 0.000001;
#endif
}

static void *malloc_aligned(size_t size) 
{
  void *ptr = (void *)NULL;
  const size_t align = 16;
  int res;
  

#if defined (__APPLE__)
  /* 
     presumably malloc on MACs always returns 
     a 16-byte aligned pointer
  */

  ptr = malloc(size);
  
  if(ptr == (void*)NULL) 
   assert(0);

#else
  res = posix_memalign( &ptr, align, size );

  if(res != 0) 
    assert(0);
#endif 
   
  return ptr;
}

static void coreGTRGAMMA(const int upper, double *d1, double *d2, double *sumtable, double *EIGN, double *gammaRates, double lz, int *wrptr)
{
  double   
    *sum, 
    diagptable0[16] __attribute__ ((aligned (16))),
    diagptable1[16] __attribute__ ((aligned (16))),
    diagptable2[16] __attribute__ ((aligned (16)));    
  int     i, j, l;
  double  dlnLdlz = 0;
  double d2lnLdlz2 = 0;
  double ki, kisqr;
  double tmp;
  double inv_Li, dlnLidlz, d2lnLidlz2;

  for(i = 0; i < 4; i++)
    {
      ki = gammaRates[i];
      kisqr = ki * ki;
      
      diagptable0[i * 4] = 1.0;
      diagptable1[i * 4] = 0.0;
      diagptable2[i * 4] = 0.0;

      for(l = 1; l < 4; l++)
	{
	  diagptable0[i * 4 + l] = exp(EIGN[l-1] * ki * lz);
	  diagptable1[i * 4 + l] = EIGN[l-1] * ki;
	  diagptable2[i * 4 + l] = EIGN[l-1] * EIGN[l-1] * kisqr;
	}
    } 

  for (i = 0; i < upper; i++)
    { 
      __m128d a0 = _mm_setzero_pd();
      __m128d a1 = _mm_setzero_pd();
      __m128d a2 = _mm_setzero_pd();

      sum = &sumtable[i * 16];         

      for(j = 0; j < 4; j++)
	{	 	  	
	  double 	   
	    *d0 = &diagptable0[j * 4],
	    *d1 = &diagptable1[j * 4],
	    *d2 = &diagptable2[j * 4];
  	 	 
	  for(l = 0; l < 4; l+=2)
	    {
	      __m128d tmpv = _mm_mul_pd(_mm_load_pd(&d0[l]), _mm_load_pd(&sum[j * 4 + l]));
	      a0 = _mm_add_pd(a0, tmpv);
	      a1 = _mm_add_pd(a1, _mm_mul_pd(tmpv, _mm_load_pd(&d1[l])));
	      a2 = _mm_add_pd(a2, _mm_mul_pd(tmpv, _mm_load_pd(&d2[l])));
	    }	 	  
	}

      a0 = _mm_hadd_pd(a0, a0);
      a1 = _mm_hadd_pd(a1, a1);
      a2 = _mm_hadd_pd(a2, a2);

      _mm_storel_pd(&inv_Li, a0);
      inv_Li = 1.0 / inv_Li;

      _mm_storel_pd(&dlnLidlz, a1);
      _mm_storel_pd(&d2lnLidlz2, a2);                
     

      dlnLidlz   *= inv_Li;
      d2lnLidlz2 *= inv_Li;     

      dlnLdlz   += wrptr[i] * dlnLidlz;
      d2lnLdlz2 += wrptr[i] * (d2lnLidlz2 - dlnLidlz * dlnLidlz);
    }

  *d1 =  dlnLdlz;
  *d2 = d2lnLdlz2;

}

int main (int argc, char *argv[])
{
  int 
    k, 
    upper,
    *wrptr;

  size_t
    fr;
  
  double
    t,
    d1,
    d2,
    *sumtable,
    EIGN[3],
    gammaRates[4],
    lz;

  FILE 
    *f = fopen("checkpointLarge", "r");
      
  fr = fread(&upper, sizeof(int), 1, f);

  sumtable = (double *)malloc_aligned(upper * 16 * sizeof(double));
  wrptr = (int*)malloc(upper * sizeof(int));

  fr = fread(sumtable, sizeof(double), upper * 16, f);
  fr = fread(EIGN, sizeof(double), 3, f);
  fr = fread(gammaRates, sizeof(double), 4, f);
  fr = fread(&lz, sizeof(double), 1, f);
  fr = fread(wrptr, sizeof(int), upper, f);
  
  fclose(f);

#ifdef _CATCH
#define MM_DAZ_ON    0x0040
  _mm_setcsr( _mm_getcsr() | (_MM_FLUSH_ZERO_ON | MM_DAZ_ON)); 
  /*_mm_setcsr( _mm_getcsr() | _MM_FLUSH_ZERO_ON);*/
#endif

#define ITS 1000

  t = gettime();

  for(k = 0; k < ITS; k++)
    coreGTRGAMMA(upper, &d1, &d2, sumtable, EIGN, gammaRates, lz, wrptr);
  
  printf("Large: %f\n", gettime() - t);

  f = fopen("checkpointSmall", "r");

  fr = fread(&upper, sizeof(int), 1, f);
  fr = fread(sumtable, sizeof(double), upper * 16, f);
  fr = fread(EIGN, sizeof(double), 3, f);
  fr = fread(gammaRates, sizeof(double), 4, f);
  fr = fread(&lz, sizeof(double), 1, f);
  fr = fread(wrptr, sizeof(int), upper, f);
  
  fclose(f);
  
  t = gettime();
  
  for(k = 0; k < ITS; k++)    
    coreGTRGAMMA(upper, &d1, &d2, sumtable, EIGN, gammaRates, lz, wrptr);    

  printf("Small: %f\n", gettime() - t);

  return 0;
}
