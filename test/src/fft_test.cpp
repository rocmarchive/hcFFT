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

#include "include/hcfftlib.h"
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <hc.hpp>
#include <iostream>
#include <map>
#include <stdio.h>
#include <stdlib.h>

#define PRINT 0

int main(int argc, char* argv[]) {
  FFTPlan plan;
  const hcfftDim dimension = HCFFT_2D;
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftResLocation location = HCFFT_OUTOFPLACE;
  hcfftResTransposed transposeType = HCFFT_NOTRANSPOSE;
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* ipStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* opStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  hcfftPlanHandle planhandle;
  hcfftPrecision precision = HCFFT_SINGLE;
  size_t N1, N2;
  N1 = argc > 1 ? atoi(argv[1]) : 1024;
  N2 = argc > 2 ? atoi(argv[2]) : 1024;
  length[0] = N1;
  length[1] = N2;
  ipStrides[0] = 1;
  ipStrides[1] = length[0];
  opStrides[0] = 1;
  opStrides[1] = 1 + length[0] / 2;
  size_t ipDistance = length[1] * length[0];
  size_t opDistance = length[1] * (1 + length[0] / 2);
  int realsize, cmplexsize;
  realsize = length[0] * length[1];
  cmplexsize = length[1] * (1 + (length[0] / 2)) * 2;
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  std::wcout << "Size: " << accs.size() << std::endl;

  if (accs.size() == 0) {
    std::wcout << "There is no acclerator!\n";
    // Since this case is to test on GPU device, skip if there is CPU only
    return 0;
  }

  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  std::for_each(accs.begin(), accs.end(), [&](hc::accelerator acc) {
    std::wcout << "New accelerator: " << acc.get_description() << std::endl;
    std::wcout << "device_path = " << acc.get_device_path() << std::endl;
    std::wcout << "version = " << (acc.get_version() >> 16) << '.'
               << (acc.get_version() & 0xFFFF) << std::endl;
    std::wcout << "dedicated_memory = " << acc.get_dedicated_memory() << " KB"
               << std::endl;
    std::wcout << "doubles = "
               << ((acc.get_supports_double_precision()) ? "true" : "false")
               << std::endl;
    std::wcout << "limited_doubles = "
               << ((acc.get_supports_limited_double_precision()) ? "true"
                                                                 : "false")
               << std::endl;
    std::wcout << "has_display = "
               << ((acc.get_has_display()) ? "true" : "false") << std::endl;
    std::wcout << "is_emulated = "
               << ((acc.get_is_emulated()) ? "true" : "false") << std::endl;
    std::wcout << "is_debug = " << ((acc.get_is_debug()) ? "true" : "false")
               << std::endl;
    std::cout << std::endl;
  });
  // Initialize host variables ----------------------------------------------
  float* ipHost = (float*)calloc(realsize, sizeof(float));
  float* ipzHost = (float*)calloc(realsize, sizeof(float));
  float* opHost = (float*)calloc(cmplexsize, sizeof(float));
  printf("ip\n");

  for (int i = 0; i < N2; i++) {
    for (int j = 0; j < N1; j++) {
      ipHost[i * N1 + j] = i * N1 + j + 1;
    }
  }

  float* ipDev = (float*)am_alloc(realsize * sizeof(float), accs[1], 0);
  float* ipzDev = (float*)am_alloc(realsize * sizeof(float), accs[1], 0);
  float* opDev = (float*)am_alloc(cmplexsize * sizeof(float), accs[1], 0);
  // Copy input contents to device from host
  accl_view.copy(ipHost, ipDev, realsize * sizeof(float));
  accl_view.copy(ipzHost, ipzDev, realsize * sizeof(float));
  accl_view.copy(opHost, opDev, cmplexsize * sizeof(float));
  hcfftLibType libtype = HCFFT_R2CD2Z;
  hcfftStatus status = plan.hcfftCreateDefaultPlan(
      &planhandle, dimension, length, dir, precision, libtype);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " Create plan error " << std::endl;
  }

  status = plan.hcfftSetAcclView(planhandle, accs[1].create_view());

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set accleration view error " << std::endl;
  }

  status = plan.hcfftSetPlanPrecision(planhandle, precision);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set plan error " << std::endl;
  }

  status = plan.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set hcfftSetPlanTransposeResult error " << std::endl;
  }

  status = plan.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set result error " << std::endl;
  }

  status = plan.hcfftSetPlanInStride(planhandle, dimension, ipStrides);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " hcfftSetPlanInStride error " << std::endl;
  }

  status = plan.hcfftSetPlanOutStride(planhandle, dimension, opStrides);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << "hcfftSetPlanOutStride error " << std::endl;
  }

  status = plan.hcfftSetPlanDistance(planhandle, ipDistance, opDistance);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << "hcfftSetPlanDistance error " << std::endl;
  }

  /*---------------------R2C--------------------------------------*/
  status =
      plan.hcfftSetLayout(planhandle, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set layout error " << std::endl;
  }

  status = plan.hcfftBakePlan(planhandle);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " bake plan error " << std::endl;
  }

  std::cout << " Starting R2C " << std::endl;
  status =
      plan.hcfftEnqueueTransform<float>(planhandle, dir, ipDev, opDev, NULL);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " Transform error " << std::endl;
  }

  std::cout << " R2C done " << std::endl;
  // Copy Device output  contents back to host
  accl_view.copy(opDev, opHost, cmplexsize * sizeof(float));
  status = plan.hcfftDestroyPlan(&planhandle);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " destroy plan error " << std::endl;
  }

#if PRINT
  printf("r2c\n");

  /* Print Output */
  for (int i = 0; i < N2; i++) {
    for (int j = 0; j < (N1 / 2 + 1) * 2; j++) {
      printf("%lf\n", opHost[i * (N1 / 2 + 1) * 2 + j]);
    }
  }

#endif
  /*---------------------C2R---------------------------------------*/
  FFTPlan plan1;
  libtype = HCFFT_C2RZ2D;
  dir = HCFFT_BACKWARD;
  ipStrides[0] = 1;
  ipStrides[1] = 1 + length[0] / 2;
  opStrides[0] = 1;
  opStrides[1] = length[0];
  ipDistance = length[1] * (1 + length[0] / 2);
  opDistance = length[0] * length[1];
  status = plan1.hcfftCreateDefaultPlan(&planhandle, dimension, length, dir,
                                        precision, libtype);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " Create plan error " << std::endl;
  }

  status = plan1.hcfftSetAcclView(planhandle, accs[1].create_view());

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set accleration view error " << std::endl;
  }

  status = plan1.hcfftSetPlanPrecision(planhandle, precision);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set plan error " << std::endl;
  }

  status = plan1.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set hcfftSetPlanTransposeResult error " << std::endl;
  }

  status = plan1.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set result error " << std::endl;
  }

  status =
      plan1.hcfftSetLayout(planhandle, HCFFT_HERMITIAN_INTERLEAVED, HCFFT_REAL);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " set layout error " << std::endl;
  }

  status = plan1.hcfftSetPlanInStride(planhandle, dimension, ipStrides);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " hcfftSetPlanInStride error " << std::endl;
  }

  status = plan1.hcfftSetPlanOutStride(planhandle, dimension, opStrides);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << "hcfftSetPlanOutStride error " << std::endl;
  }

  status = plan1.hcfftSetPlanDistance(planhandle, ipDistance, opDistance);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << "hcfftSetPlanDistance error " << std::endl;
  }

  status = plan1.hcfftSetPlanScale(planhandle, dir, 1.0);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " setplan scale error " << std::endl;
  }

  status = plan1.hcfftBakePlan(planhandle);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " bake plan error " << std::endl;
  }

  std::cout << " Starting C2R " << std::endl;
  status =
      plan1.hcfftEnqueueTransform<float>(planhandle, dir, opDev, ipzDev, NULL);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " Transform error " << std::endl;
  }

  std::cout << " C2R done " << std::endl;
  // Copy Device output  contents back to host
  accl_view.copy(ipzDev, ipzHost, realsize * sizeof(float));
#if PRINT

  /* Print Output */
  for (int i = 0; i < N2; i++) {
    for (int j = 0; j < N1; j++) {
      std::cout << " ipzHost[" << i * N1 + j << "] " << ipzHost[i * N1 + j]
                << std::endl;
    }
  }

#endif
  std::cout << " Comparing results " << std::endl;

  for (int i = 0; i < N2; i++) {
    for (int j = 0; j < N1; j++) {
      if ((round(ipzHost[i * N1 + j]) != (ipHost[i * N1 + j]) * N1 * N2) ||
          std::isnan(ipzHost[i * N1 + j])) {
        std::cout << " Mismatch at  " << i * N1 + j << " input "
             << ipHost[i * N1 + j] << " amp " << round(ipzHost[i * N1 + j])
             << std::endl;
        std::cout << " TEST FAILED " << std::endl;
        exit(0);
      }
    }
  }

  std::cout << " TEST PASSED " << std::endl;
  status = plan1.hcfftDestroyPlan(&planhandle);

  if (status != HCFFT_SUCCEEDS) {
    std::cout << " destroy plan error " << std::endl;
  }

  hc::am_free(ipDev);
  hc::am_free(ipzDev);
  hc::am_free(opDev);
  free(ipHost);
  free(ipzHost);
  free(opHost);
}
