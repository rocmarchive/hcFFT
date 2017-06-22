#include <iostream>
#include <cstdlib>
#include "hcfft.h"
#include "hc_am.hpp"
#include "hcfftlib.h"

int main(int argc, char* argv[]) {
  int N = argc > 1 ? atoi(argv[1]) : 1024;
  hcfftHandle plan;
  hcfftResult status  = hcfftPlan1d(&plan, N, HCFFT_C2C);
  assert(status == HCFFT_SUCCESS);
  int hSize = N;
  hcfftComplex* input = (hcfftComplex*)calloc(hSize, sizeof(hcfftComplex));
  hcfftComplex* output = (hcfftComplex*)calloc(hSize, sizeof(hcfftComplex));
  int seed = 123456789;
  srand(seed);

  std::cout<<"Input: "<<std::endl;
  // Populate the input
  for(int i = 0; i < hSize ; i++) {
    input[i].x = rand();
    input[i].y = rand();
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftComplex* idata = hc::am_alloc(hSize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftComplex) * hSize);
  hcfftComplex* odata = hc::am_alloc(hSize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftComplex) * hSize);
  status = hcfftExecC2C(plan, idata, odata, HCFFT_FORWARD);
  assert(status == HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftComplex) * hSize);  
  status =  hcfftDestroy(plan);
  assert(status == HCFFT_SUCCESS);

  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
