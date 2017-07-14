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
#include "./helper_functions.h"

TEST(hcfft_3D_transform_test, func_correct_3D_transform_D2Z) {
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 4;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 4;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 4;
  hcfftHandle plan;
  hcfftResult status = hcfftPlan3d(&plan, N1, N2, N3, HCFFT_D2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N1 * N2 * N3;
  int Csize = N3 * N2 * (1 + N1 / 2);
  hcfftDoubleReal* input =
      (hcfftDoubleReal*)malloc(Rsize * sizeof(hcfftDoubleReal));
  hcfftDoubleComplex* output =
      (hcfftDoubleComplex*)malloc(Csize * sizeof(hcfftDoubleComplex));

  // Populate the input
  for (int i = 0; i < Rsize; i++) {
    input[i] = i % 8;
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftDoubleReal* idata =
      hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftDoubleReal) * Rsize);
  hcfftDoubleComplex* odata =
      hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftDoubleComplex) * Csize);
  status = hcfftExecD2Z(plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftDoubleComplex) * Csize);
  status = hcfftDestroy(plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  // FFTW work flow
  // input output arrays
  double* in;
  fftw_complex* out;
  fftw_plan p;
  in = (double*)fftw_malloc(sizeof(double) * Rsize);
  out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Csize);

  // Populate inputs
  for (int i = 0; i < Rsize; i++) {
    in[i] = input[i];
  }

  // 3D forward plan
  p = fftw_plan_dft_r2c_3d(N3, N2, N1, in, out, FFTW_ESTIMATE | FFTW_R2HC);

  // Execute R2C

  // Check RMSE: If fails move on to pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftw_complex, hcfftDoubleComplex>(out, output,
                                                                 Csize)) {
    fftw_execute(p);
    // Check Real Outputs
    for (int i = 0; i < Csize; i++) {
      EXPECT_NEAR(out[i][0], output[i].x, 0.1);
    }
    // Check Imaginary Outputs
    for (int i = 0; i < Csize; i++) {
      EXPECT_NEAR(out[i][1], output[i].y, 0.1);
    }
  }
  // Free up resources
  fftw_destroy_plan(p);
  fftw_free(in);
  fftw_free(out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

