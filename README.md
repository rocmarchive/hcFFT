## A. Introduction: ##

This repository hosts the HCC based FFT Library, that targets GPU acceleration of FFT routines on AMD devices. To know what HCC compiler features, refer [here](https://bitbucket.org/multicoreware/hcc/wiki/Home).

The following are the sub-routines that are implemented

1. R2C : Transforms Real valued input in Time domain to Complex valued output in Frequency domain.
2. C2R : Transforms Complex valued input in Frequency domain to Real valued output in Real domain.
3. C2C : Transforms Complex valued input in Frequency domain to Complex valued output in Real domain or vice versa

To know more, go through the [Documentation](http://hcfft.readthedocs.org/en/latest/)


## B. Key Features ##

* Support 1D, 2D and 3D Fast Fourier Transforms
* Supports R2C, C2R, C2C, D2Z, Z2D and Z2Z Transforms
* Support Out-Of-Place data storage
* Ability to Choose desired target accelerator
* Single and Double precision

## C. Prerequisites ##

* Refer Prerequisites section [here](http://hcfft.readthedocs.org/en/latest/#prerequisites)

## D. Tested Environment so far

* Refer Tested environments enumerated [here](http://hcfft.readthedocs.org/en/latest/#tested-environments)


## E. Installation

* Follow installation steps as described [here](http://hcfft.readthedocs.org/en/latest/#installation-steps)


## F. Unit testing

* Follow testing procedures as explained [here](http://hcfft.readthedocs.org/en/latest/#unit-testing)

## G. API reference

* The Specification of API's supported along with description  can be found [here](http://hcfft.readthedocs.org/en/latest/#hcfft-api-reference)

## H. Example Code

FFT 1D R2C example: 

file: hcfft_1D_R2C.cpp

```
#!c++

#include "hcfft.h"

int main(int argc, char *argv[])
{
  int N = atoi(argv[1]);

  // HCFFT work flow
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N, HCFFT_R2C);
  assert(status == HCFFT_SUCCESS);
  int Rsize = N;
  int Csize = (N / 2) + 1;
  hcfftReal *input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = rand();
  }
  hcfftComplex *output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));

  std::vector<accelerator> accs = accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");

  hcfftReal *idata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftReal) * Rsize);

  hcfftComplex *odata = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftComplex) * Csize);

  status = hcfftExecR2C(*plan, idata, odata);
  assert(status == HCFFT_SUCCESS);

  hc::am_copy(output, odata, sizeof(hcfftComplex) * Csize);

  status =  hcfftDestroy(*plan);
  assert(status == HCFFT_SUCCESS);

  free(input);
  free(output);

  hc::am_free(idata);
  hc::am_free(odata);
}
