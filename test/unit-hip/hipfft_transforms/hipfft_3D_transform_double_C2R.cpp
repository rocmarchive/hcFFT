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
#include <fftw3.h>
#include "hip/hip_runtime.h"

TEST(hipfft_3D_transform_test, func_correct_3D_transform_Z2D_RTT) {
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 4;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 4;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 4;

  // HIPFFT work flow
  hipfftHandle plan;
  hipfftResult status = hipfftPlan3d(&plan, N1, N2, N3, HIPFFT_D2Z);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  int Rsize = N1 * N2 * N3;
  int Csize = N1 * N2 * (1 + N3 / 2);
  hipfftDoubleReal* inputR2C =
      (hipfftDoubleReal*)malloc(Rsize * sizeof(hipfftDoubleReal));
  hipfftDoubleComplex* outputR2C =
      (hipfftDoubleComplex*)malloc(Csize * sizeof(hipfftDoubleComplex));

  // Populate the input
  for (int i = 0; i < Rsize; i++) {
    inputR2C[i] = i % 8;
  }

  hipfftDoubleReal* devIpR2C;
  hipfftDoubleComplex* devOpR2C;
  hipMalloc(&devIpR2C, Rsize * sizeof(hipfftDoubleReal));
  hipMemcpy(devIpR2C, inputR2C, sizeof(hipfftDoubleReal) * Rsize,
            hipMemcpyHostToDevice);
  hipMalloc(&devOpR2C, Csize * sizeof(hipfftDoubleComplex));
  hipMemcpy(devOpR2C, outputR2C, sizeof(hipfftDoubleComplex) * Csize,
            hipMemcpyHostToDevice);
  status = hipfftExecD2Z(plan, devIpR2C, devOpR2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(outputR2C, devOpR2C, sizeof(hipfftDoubleComplex) * Csize,
            hipMemcpyDeviceToHost);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  status = hipfftPlan3d(&plan, N1, N2, N3, HIPFFT_Z2D);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  hipfftDoubleComplex* inputC2R =
      (hipfftDoubleComplex*)malloc(Csize * sizeof(hipfftDoubleComplex));
  hipfftDoubleReal* outputC2R =
      (hipfftDoubleReal*)malloc(Rsize * sizeof(hipfftDoubleReal));

  // Populate the input
  for (int i = 0; i < Csize; i++) {
    inputC2R[i].x = outputR2C[i].x;
    inputC2R[i].y = outputR2C[i].y;
  }

  hipfftDoubleComplex* devIpC2R;
  hipfftDoubleReal* devOpC2R;
  hipMalloc(&devIpC2R, Csize * sizeof(hipfftDoubleComplex));
  hipMemcpy(devIpC2R, inputC2R, sizeof(hipfftDoubleComplex) * Csize,
            hipMemcpyHostToDevice);
  hipMalloc(&devOpC2R, Rsize * sizeof(hipfftDoubleReal));
  hipMemcpy(devOpC2R, outputC2R, sizeof(hipfftDoubleReal) * Rsize,
            hipMemcpyHostToDevice);
  status = hipfftExecZ2D(plan, devIpC2R, devOpC2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(outputC2R, devOpC2R, sizeof(hipfftDoubleReal) * Rsize,
            hipMemcpyDeviceToHost);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  // Check Real Inputs and  Outputs
  for (int i = 0; i < Rsize; i++) {
    EXPECT_NEAR(inputR2C[i], outputC2R[i] / Rsize, 0.1);
  }

  // Free up resources
  free(inputC2R);
  free(outputC2R);
  free(outputR2C);
  free(inputR2C);
  hipFree(devIpR2C);
  hipFree(devIpC2R);
  hipFree(devOpR2C);
  hipFree(devOpC2R);
}

