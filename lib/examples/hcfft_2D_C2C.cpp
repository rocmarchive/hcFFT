#include "hcfft.h"

int main(int argc, char *argv[])
{
  int N1 = atoi(argv[1]);
  int N2 = atoi(argv[2]);

  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, N1, N2, HCFFT_C2C);
  assert(status == HCFFT_SUCCESS);
  int hSize = N1 * N2;
  hcfftComplex *input = (hcfftComplex*)calloc(hSize, sizeof(hcfftComplex));
  hcfftComplex *output = (hcfftComplex*)calloc(hSize, sizeof(hcfftComplex));
  int seed = 123456789;
  srand(seed);
  // Populate the input
  for(int i = 0; i < hSize ; i++)
  {
    input[i].x = rand();
    input[i].y = rand();
  }

  std::vector<accelerator> accs = accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");

  hcfftComplex *idata = hc::am_alloc(hSize * sizeof(hcfftComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftComplex) * hSize);

  hcfftComplex *odata = hc::am_alloc(hSize * sizeof(hcfftComplex), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftComplex) * hSize);

  status = hcfftExecC2C(*plan, idata, odata, HCFFT_FORWARD);
  assert(status == HCFFT_SUCCESS);

  hc::am_copy(output, odata, sizeof(hcfftComplex) * hSize);

  status =  hcfftDestroy(*plan);
  assert(status == HCFFT_SUCCESS);

  free(input);
  free(output);

  hc::am_free(idata);
  hc::am_free(odata);
}
