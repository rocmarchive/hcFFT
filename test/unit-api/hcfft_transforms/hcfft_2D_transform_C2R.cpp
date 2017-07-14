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

#include "include/hcfft.h"
#include "../gtest/gtest.h"
#include <fftw3.h>
#include <hc_am.hpp>
#include "include/hcfftlib.h"

TEST(hcfft_2D_transform_test, func_correct_2D_transform_C2R_RTT) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle plan;
  hcfftResult status = hcfftPlan2d(&plan, N1, N2, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N1 * N2;
  int Csize = N2 * (1 + N1 / 2);
  hcfftReal* inputR2C = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  hcfftComplex* outputR2C = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  
  // Populate the input
  for (int i = 0; i < Rsize; i++) {
    inputR2C[i] = i % 8;
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftReal* devIpR2C = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(inputR2C, devIpR2C, sizeof(hcfftReal) * Rsize);
  hcfftComplex* devOpR2C =
      hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(outputR2C, devOpR2C, sizeof(hcfftComplex) * Csize);
  status = hcfftExecR2C(plan, devIpR2C, devOpR2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(devOpR2C, outputR2C, sizeof(hcfftComplex) * Csize);
  status = hcfftDestroy(plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  // plan = NULL;
  status = hcfftPlan2d(&plan, N1, N2, HCFFT_C2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  hcfftComplex* inputC2R = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  hcfftReal* outputC2R = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));

  // Populate the input of C2R with output ofo R2C
  for (int i = 0; i < Csize; i++) {
    inputC2R[i].x = outputR2C[i].x;
    inputC2R[i].y = outputR2C[i].y;
  }

  hcfftComplex* devIpC2R =
      hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(inputC2R, devIpC2R, sizeof(hcfftComplex) * Csize);
  hcfftReal* devOpC2R = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(outputC2R, devOpC2R, sizeof(hcfftReal) * Rsize);
  status = hcfftExecC2R(plan, devIpC2R, devOpC2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(devOpC2R, outputC2R, sizeof(hcfftReal) * Rsize);
  status = hcfftDestroy(plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  // Check Real Outputs
  for (int i = 0; i < Rsize; i++) {
    EXPECT_NEAR(outputC2R[i] / (Rsize), inputR2C[i], 1);
  }

  // Free up resources
  free(inputR2C);
  free(outputR2C);
  free(inputC2R);
  free(outputC2R);
  hc::am_free(devOpC2R);
  hc::am_free(devOpR2C);
  hc::am_free(devIpC2R);
  hc::am_free(devIpR2C);
}

