#include <iostream>
#include <cstdlib>
#include "hcfft.h"
#include "hc_am.hpp"
#include "hcfftlib.h"

int main(int argc, char* argv[]) {
  int N1 = argc > 1 ? atoi(argv[1]) : 1024;
  int N2 = argc > 2 ? atoi(argv[2]) : 1024;
  hcfftHandle plan;
  hcfftResult status  = hcfftPlan2d(&plan, N1, N2,  HCFFT_R2C);
  assert(status == HCFFT_SUCCESS);
  int Rsize = N2 * N1;
  int Csize = N2 * (1 + N1 / 2);
  hcfftReal* input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = rand();
  }

  hcfftComplex* output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftReal* idata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftReal) * Rsize);
  hcfftComplex* odata = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(output,  odata, sizeof(hcfftComplex) * Csize);
  status = hcfftExecR2C(plan, idata, odata);
  assert(status == HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftComplex) * Csize);
  status =  hcfftDestroy(plan);
  assert(status == HCFFT_SUCCESS);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}


