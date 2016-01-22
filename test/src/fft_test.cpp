#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include "hcfftlib.h"
#include <dlfcn.h>
#include <map>
#include <hc.hpp>
#include <cfloat>
#include <stdio.h>
#include <cmath>

#define PRINT 0

using namespace hc;
using namespace std;

int main(int argc, char* argv[]) {
  FFTPlan plan;
  hcfftDim dimension = HCFFT_2D;
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftResLocation location = HCFFT_OUTOFPLACE;
  hcfftResTransposed transposeType = HCFFT_NOTRANSPOSE;
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  hcfftPlanHandle planhandle;
  hcfftPrecision precision = HCFFT_SINGLE;
  size_t N1, N2;
  N1 = atoi(argv[1]);
  N2 = atoi(argv[2]);
  length[0] = N1;
  length[1] = N2;
  int realsize, cmplexsize;
  realsize = length[0] * length[1];
  cmplexsize = length[1] * (1 + (length[0] / 2)) * 2;

  std::vector<accelerator> accs = accelerator::get_all();
  std::wcout << "Size: " << accs.size() << std::endl;
  if(accs.size() == 0) {
    std::wcout << "There is no acclerator!\n";
    // Since this case is to test on GPU device, skip if there is CPU only
    return 0;
  }
  assert(accs.size() && "Number of Accelerators == 0!");
  std::for_each(accs.begin(), accs.end(), [&] (accelerator acc)
  {
    std::wcout << "New accelerator: " << acc.get_description() << std::endl;
    std::wcout << "device_path = " << acc.get_device_path() << std::endl;
    std::wcout << "version = " << (acc.get_version() >> 16) << '.' << (acc.get_version() & 0xFFFF) << std::endl;
    std::wcout << "dedicated_memory = " << acc.get_dedicated_memory() << " KB" << std::endl;
    std::wcout << "doubles = " << ((acc.get_supports_double_precision()) ? "true" : "false") << std::endl;
    std::wcout << "limited_doubles = " << ((acc.get_supports_limited_double_precision()) ? "true" : "false") << std::endl;
    std::wcout << "has_display = " << ((acc.get_has_display()) ? "true" : "false") << std::endl;
    std::wcout << "is_emulated = " << ((acc.get_is_emulated()) ? "true" : "false") << std::endl;
    std::wcout << "is_debug = " << ((acc.get_is_debug()) ? "true" : "false") << std::endl;
    std::cout << std::endl;
  });

  // Initialize host variables ----------------------------------------------
  float* ipHost = (float*)calloc(realsize, sizeof(float));
  float* ipzHost = (float*)calloc(realsize, sizeof(float));
  float* opHost = (float*)calloc(cmplexsize, sizeof(float));

  for(int  i = 0; i < realsize ; i++) {
    ipHost[i] = i + 1;
  }

  float* ipDev = (float*)am_alloc(realsize * sizeof(float), accs[1], 0);
  float* ipzDev = (float*)am_alloc(realsize * sizeof(float), accs[1], 0);
  float* opDev = (float*)am_alloc(cmplexsize * sizeof(float), accs[1], 0);

  // Copy input contents to device from host
  hc::am_copy(ipDev, ipHost, realsize * sizeof(float));
  hc::am_copy(ipzDev, ipzHost, realsize * sizeof(float));
  hc::am_copy(opDev, opHost, cmplexsize * sizeof(float));

  hcfftStatus status = plan.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir, accs[1]);
  status = plan.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set plan error " << endl;
  }

  status = plan.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set result error " << endl;
  }

  /*---------------------R2C--------------------------------------*/
  status = plan.hcfftSetLayout(planhandle, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set layout error " << endl;
  }

  status = plan.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCEEDS) {
    cout << " bake plan error " << endl;
  }

  plan.hcfftEnqueueTransform(planhandle, dir, ipDev, opDev, NULL);

  // Copy Device output  contents back to host
  hc::am_copy(opHost, opDev, cmplexsize * sizeof(float));

  status = plan.hcfftDestroyPlan(&planhandle);
#if PRINT

  /* Print Output */
  for (int i = 0; i < cmplexsize; i++) {
    std::cout << " opHost[" << i << "] " << opHost[i] << std::endl;
  }

#endif
  /*---------------------C2R---------------------------------------*/

  FFTPlan plan1;
  dir = HCFFT_BACKWARD;
  status = plan1.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir, accs[1]);
  status = plan1.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set plan error " << endl;
  }

  status = plan1.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan1.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set result error " << endl;
  }

  status = plan1.hcfftSetLayout(planhandle, HCFFT_HERMITIAN_INTERLEAVED, HCFFT_REAL);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set layout error " << endl;
  }

  status = plan1.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCEEDS) {
    cout << " bake plan error " << endl;
  }

  plan1.hcfftEnqueueTransform(planhandle, dir, opDev, ipzDev, NULL);

  // Copy Device output  contents back to host
  hc::am_copy(ipzHost, ipzDev, realsize * sizeof(float));
#if PRINT

  /* Print Output */
  for (int i = 0; i < realsize; i++) {
    std::cout << " ipzHost[" << i << "] " << ipzHost[i] << std::endl;
  }

#endif

  for(int  i = 0; i < realsize; i++)
    if((round(ipzHost[i]) != ipHost[i]) && isnan(ipzHost[i])) {
      cout << " Mismatch at  " << i << " input " << ipHost[i] << " amp " << round(ipzHost[i]) << endl;
      cout << " TEST FAILED " << std::endl;
      exit(0);
    }

  cout << " TEST PASSED " << std::endl;
  status = plan1.hcfftDestroyPlan(&planhandle);

  hc::am_free(ipDev);
  hc::am_free(ipzDev);
  hc::am_free(opDev);

  free(ipHost);
  free(ipzHost);
  free(opHost);
}
