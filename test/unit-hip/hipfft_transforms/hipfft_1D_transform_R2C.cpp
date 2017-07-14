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
#include "./helper_functions.h"
#include "hip/hip_runtime.h"

TEST(hipfft_1D_transform_test, func_correct_1D_transform_R2C) {
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;

  // HIPFFT work flow
  hipfftHandle plan;
  hipfftResult status = hipfftPlan1d(&plan, N1, HIPFFT_R2C, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  int Rsize = N1;
  int Csize = (N1 / 2) + 1;
  hipfftReal* input = (hipfftReal*)calloc(Rsize, sizeof(hipfftReal));
  hipfftComplex* output = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));

  // Populate the input
  for (int i = 0; i < Rsize; i++) {
    input[i] = i % 8;
  }

  hipfftReal* idata;
  hipfftComplex* odata;
  hipMalloc(&idata, Rsize * sizeof(hipfftReal));
  hipMemcpy(idata, input, sizeof(hipfftReal) * Rsize, hipMemcpyHostToDevice);
  hipMalloc(&odata, Csize * sizeof(hipfftComplex));
  hipMemcpy(odata, output, sizeof(hipfftComplex) * Csize,
            hipMemcpyHostToDevice);
  status = hipfftExecR2C(plan, idata, odata);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(output, odata, sizeof(hipfftComplex) * Csize,
            hipMemcpyDeviceToHost);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  // FFTW work flow
  // input output arrays
  float* in;
  fftwf_complex* out;
  int lengths[1] = {Rsize};
  fftwf_plan p;
  in = (float*)fftwf_malloc(sizeof(float) * Rsize);
  out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Csize);

  // Populate inputs
  for (int i = 0; i < Rsize; i++) {
    in[i] = input[i];
  }

  // 1D forward plan
  p = fftwf_plan_many_dft_r2c(1, lengths, 1, in, NULL, 1, 0, out, NULL, 1, 0,
                              FFTW_ESTIMATE | FFTW_R2HC);

  // Execute R2C
  fftwf_execute(p);

  // Check RMSE: If fails go for pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftwf_complex, hipfftComplex>(out, output,
                                                             Csize)) {
    // Check Real Outputs
    for (int i = 0; i < Csize; i++) {
      EXPECT_NEAR(out[i][0], output[i].x, 0.01);
    }
    // Check Imaginary Outputs
    for (int i = 0; i < Csize; i++) {
      EXPECT_NEAR(out[i][1], output[i].y, 0.01);
    }
  }

  // Free up resources
  fftwf_destroy_plan(p);
  fftwf_free(in);
  fftwf_free(out);
  free(input);
  free(output);
  hipFree(idata);
  hipFree(odata);
}

