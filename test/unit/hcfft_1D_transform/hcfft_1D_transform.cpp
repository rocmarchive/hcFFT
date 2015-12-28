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
