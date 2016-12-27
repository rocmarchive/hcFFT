#include "hipfft.h"
#include "../gtest/gtest.h"
#include "hip/hip_runtime_api.h"

#define VECTOR_SIZE 256

TEST(hipfft_Create_Destroy_Plan, create_destroy_1D_plan_R2C ) {
  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan1d(&plan, VECTOR_SIZE, HIPFFT_R2C, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}
/*
TEST(hipfft_Create_Destroy_Plan, create_destroy_2D_plan_R2C ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan2d(&plan, VECTOR_SIZE, VECTOR_SIZE,  HIPFFT_R2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_3D_plan_R2C ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan3d(&plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_R2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_1D_plan_C2R ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan1d(&plan, VECTOR_SIZE, HIPFFT_C2R, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_2D_plan_C2R ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan2d(&plan, VECTOR_SIZE, VECTOR_SIZE,  HIPFFT_C2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_3D_plan_C2R ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan3d(&plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_C2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_1D_plan_C2C ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan1d(&plan, VECTOR_SIZE, HIPFFT_C2C, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_2D_plan_C2C ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan2d(&plan, VECTOR_SIZE, VECTOR_SIZE,  HIPFFT_C2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_3D_plan_C2C ) {
  hipfftHandle plan;// = NULL;
  hipfftResult status  = hipfftPlan3d(&plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_C2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}
*/
