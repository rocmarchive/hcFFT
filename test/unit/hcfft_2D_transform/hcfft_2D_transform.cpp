#include "hcfft.h"
#include "gtest/gtest.h"
#define VECTOR_SIZE 256

  
TEST(hcfft_2D_transform_test, func_correct_2D_transform_R2C ) {
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, VECTOR_SIZE, VECTOR_SIZE,  HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = VECTOR_SIZE * VECTOR_SIZE;
  int Csize = VECTOR_SIZE * (1 + VECTOR_SIZE / 2);
  hcfftReal *input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  hcfftComplex *output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  Concurrency::array_view<hcfftReal, 1> idata(Rsize, input);
  Concurrency::array_view<hcfftComplex, 1> odata(Csize, output);
  status = hcfftExecR2C(*plan, &idata, &odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
}
