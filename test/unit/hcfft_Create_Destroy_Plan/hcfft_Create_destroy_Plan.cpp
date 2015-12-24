#include "hcfft.h"
#include "gtest/gtest.h"

TEST(hcfft_Create_Destroy_Plan, create_destroy_1D_plan ) {
  hcfftHandle *plan;
  hcfftResult status  = hcfftPlan1d(plan, 1024, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //status =  hcfftDestroy(*plan);
  //EXPECT_EQ(status, HCFFT_SUCCESS);
}
