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
#include "include/hcfftlib.h"
#include <hc_am.hpp>
#include <cstdlib>
#include <iostream>

unsigned int global_seed = 100;

int main(int argc, char* argv[]) {
  int N1 = argc > 1 ? atoi(argv[1]) : 1024;
  int N2 = argc > 2 ? atoi(argv[2]) : 1024;
  hcfftHandle plan;
  hcfftResult status = hcfftPlan2d(&plan, N1, N2, HCFFT_C2R);
  assert(status == HCFFT_SUCCESS);
  int Csize = N2 * (1 + N1 / 2);
  int Rsize = N2 * N1;
  hcfftComplex* input = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  hcfftReal* output = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));

  // Populate the input
  for (int i = 0; i < Csize; i++) {
    input[i].x = rand_r(&global_seed);
    input[i].y = rand_r(&global_seed);
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();

  hcfftComplex* idata = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftComplex) * Csize);
  hcfftReal* odata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftReal) * Rsize);
  status = hcfftExecC2R(plan, idata, odata);
  assert(status == HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftReal) * Rsize);
  status = hcfftDestroy(plan);
  assert(status == HCFFT_SUCCESS);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

