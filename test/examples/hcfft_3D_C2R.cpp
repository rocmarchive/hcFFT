#include "hcfft.h"

int main(int argc, char* argv[]) {
  int N1 = argc > 1 ? atoi(argv[1]) : 1024;
  int N2 = argc > 2 ? atoi(argv[2]) : 1024;
  int N3 = argc > 3 ? atoi(argv[3]) : 1024;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan3d(plan, N1, N2, N3, HCFFT_C2R);
  assert(status == HCFFT_SUCCESS);
  int Csize = N3 * N2 * (1 + N1 / 2);
  int Rsize = N3 * N2 * N1;
  hcComplex* input = (hcComplex*)calloc(Csize, sizeof(hcComplex));
  hcfftReal* output = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    input[i].x = rand();
    input[i].y = rand();
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hcComplex* idata = hc::am_alloc(Csize * sizeof(hcComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcComplex) * Csize);
  hcfftReal* odata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftReal) * Rsize);
  status = hcfftExecC2R(*plan, idata, odata);
  assert(status == HCFFT_SUCCESS);
  hc::am_copy(output, odata, sizeof(hcfftReal) * Rsize);
  status =  hcfftDestroy(*plan);
  assert(status == HCFFT_SUCCESS);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
