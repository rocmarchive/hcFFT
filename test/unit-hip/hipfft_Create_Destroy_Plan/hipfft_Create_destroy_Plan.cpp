/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "include/hipfft.h"
#include "../gtest/gtest.h"
#include "hip/hip_runtime.h"

#define VECTOR_SIZE 256

TEST(hipfft_Create_Destroy_Plan, create_destroy_1D_plan_R2C) {
  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  hipfftHandle plan;
  hipfftResult status = hipfftPlan1d(&plan, VECTOR_SIZE, HIPFFT_R2C, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_2D_plan_R2C) {
  hipfftHandle plan;
  hipfftResult status =
      hipfftPlan2d(&plan, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_R2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_3D_plan_R2C) {
  hipfftHandle plan;
  hipfftResult status =
      hipfftPlan3d(&plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_R2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_1D_plan_C2R) {
  hipfftHandle plan;
  hipfftResult status = hipfftPlan1d(&plan, VECTOR_SIZE, HIPFFT_C2R, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_2D_plan_C2R) {
  hipfftHandle plan;
  hipfftResult status =
      hipfftPlan2d(&plan, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_C2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_3D_plan_C2R) {
  hipfftHandle plan;
  hipfftResult status =
      hipfftPlan3d(&plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_C2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_1D_plan_C2C) {
  hipfftHandle plan;
  hipfftResult status = hipfftPlan1d(&plan, VECTOR_SIZE, HIPFFT_C2C, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_2D_plan_C2C) {
  hipfftHandle plan;
  hipfftResult status =
      hipfftPlan2d(&plan, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_C2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

TEST(hipfft_Create_Destroy_Plan, create_destroy_3D_plan_C2C) {
  hipfftHandle plan;
  hipfftResult status =
      hipfftPlan3d(&plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HIPFFT_C2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
}

