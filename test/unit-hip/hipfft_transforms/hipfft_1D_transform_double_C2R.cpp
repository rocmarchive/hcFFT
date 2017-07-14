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

TEST(hipfft_1D_transform_double_test, func_correct_1D_transform_Z2D) {
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;

  // HIPFFT work flow
  hipfftHandle plan;
  hipfftResult status = hipfftPlan1d(&plan, N1, HIPFFT_Z2D, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  int Csize = (N1 / 2) + 1;
  int Rsize = N1;
  hipfftDoubleComplex* input =
      (hipfftDoubleComplex*)calloc(Csize, sizeof(hipfftDoubleComplex));
  hipfftDoubleReal* output =
      (hipfftDoubleReal*)calloc(Rsize, sizeof(hipfftDoubleReal));

  // Populate the input
  for (int i = 0; i < Csize; i++) {
    input[i].x = i % 8;
    input[i].y = i % 16;
  }

  hipfftDoubleComplex* idata;
  hipfftDoubleReal* odata;
  hipMalloc(&idata, Csize * sizeof(hipfftDoubleComplex));
  hipMemcpy(idata, input, sizeof(hipfftDoubleComplex) * Csize,
            hipMemcpyHostToDevice);
  hipMalloc(&odata, Rsize * sizeof(hipfftDoubleReal));
  hipMemcpy(odata, output, sizeof(hipfftDoubleReal) * Rsize,
            hipMemcpyHostToDevice);
  status = hipfftExecZ2D(plan, idata, odata);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(output, odata, sizeof(hipfftDoubleReal) * Rsize,
            hipMemcpyDeviceToHost);
  status = hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  // FFTW work flow
  // input output arrays
  double* fftw_out;
  fftw_complex* fftw_in;
  int lengths[1] = {Rsize};
  fftw_plan p;
  fftw_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Csize);
  fftw_out = (double*)fftw_malloc(sizeof(double) * Rsize);

  // Populate inputs
  for (int i = 0; i < Csize; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }

  // 1D forward plan
  p = fftw_plan_many_dft_c2r(1, lengths, 1, fftw_in, NULL, 1, 0, fftw_out, NULL,
                             1, 0, FFTW_ESTIMATE | FFTW_HC2R);

  // Execute C2R
  fftw_execute(p);

  // Check RMSE : IF fails go for
  if (JudgeRMSEAccuracyReal<double, hipfftDoubleReal>(fftw_out, output,
                                                      Rsize)) {
    // Check Real Outputs
    for (int i = 0; i < Rsize; i++) {
      EXPECT_NEAR(fftw_out[i], output[i], 1);
    }
  }

  // Free up resources
  fftw_destroy_plan(p);
  fftw_free(fftw_in);
  fftw_free(fftw_out);
  free(input);
  free(output);
  hipFree(idata);
  hipFree(odata);
}
