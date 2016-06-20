#include "hcfft.h"

int main(int argc, char *argv[])
{
  int N = argc > 1 ? atoi(argv[1]) : 1024;

  // HCFFT work flow
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N, HCFFT_C2R);
  assert(status == HCFFT_SUCCESS);
  int Csize = (N / 2) + 1;
  int Rsize = N;
  hcfftComplex *input = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Csize ; i++)
  {
    input[i].x = rand();
    input[i].y = rand();
  }
  hcfftReal *output = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");

  hcfftComplex *idata = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftComplex) * Csize);

  hcfftReal *odata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
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
