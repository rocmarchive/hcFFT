#include "hcfft.h"
#include "gtest/gtest.h"
#include"fftw3.h"
#define VECTOR_SIZE 1024
 
TEST(hcfft_1D_transform_test, func_correct_1D_transform_R2C ) {
  // HCFFT work flow
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, VECTOR_SIZE, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = VECTOR_SIZE;
  int Csize = VECTOR_SIZE / 2;
  hcfftReal *input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);
  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = rand();
  }
  hcfftComplex *output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  Concurrency::array_view<hcfftReal> idata(Rsize, input);
  Concurrency::array_view<hcfftComplex> odata(Csize, output);
  status = hcfftExecR2C(*plan, &idata, &odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  // FFTW work flow
  double *in;
  fftw_complex *out;
  int n = VECTOR_SIZE;
  int nc = ( n / 2 ) + 1;
  fftw_plan plan_forward;
  in = (double *)fftw_malloc ( sizeof ( double ) * n );
  out = (fftw_complex *)fftw_malloc ( sizeof ( fftw_complex ) * nc );
  // Populate the input as given to hcfft
  for(int i = 0; i < n; i++) {
    in[i] = input[i];
  }
  plan_forward = fftw_plan_dft_r2c_1d ( n, in, out, FFTW_ESTIMATE );
  fftw_execute ( plan_forward );
  fftw_destroy_plan ( plan_forward );
  //Compare the results of FFTW and HCFFT with 0.01 precision
  for(int i=0; i< nc; i++) {
    EXPECT_NEAR((float)out[i][0], odata[i].x, 0.01);
    EXPECT_NEAR((float)out[i][1], odata[i].y, 0.01);
  }

  // free up allocated resources
  fftw_free(in);
  fftw_free(out);
  free(input);
  fftw_free(output);
}

TEST(hcfft_1D_transform_test, func_correct_1D_transform_C2R ) {
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, VECTOR_SIZE, HCFFT_C2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Csize = VECTOR_SIZE / 2;
  int Rsize = VECTOR_SIZE;
  hcfftComplex *input = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  for(int i = 0; i < Csize ; i++)
  {
    input[i].x = i + 100;
    input[i].y = 0.0;
  }

  hcfftReal *output = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  Concurrency::array_view<hcfftComplex> idata(Csize, input);
  Concurrency::array_view<hcfftReal> odata(Rsize, output);
  status = hcfftExecC2R(*plan, &idata, &odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
}
