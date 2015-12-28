#include "hcfft.h"
#include "gtest/gtest.h"
#define VECTOR_SIZE 256
 
TEST(hcfft_1D_transform_test, func_correct_1D_transform_R2C ) {
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, VECTOR_SIZE, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = VECTOR_SIZE;
  int Csize = VECTOR_SIZE / 2;
  hcfftReal *input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  for(int i = 0; i < Rsize ; i++)
    input[i] = i + 1;

  hcfftComplex *output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  Concurrency::array_view<hcfftReal> idata(Rsize, input);
  Concurrency::array_view<hcfftComplex> odata(Csize, output);
  status = hcfftExecR2C(*plan, &idata, &odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
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
