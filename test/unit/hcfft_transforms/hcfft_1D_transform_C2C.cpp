#include "hcfft.h"
#include "fftw3.h"
#include "../gtest/gtest.h"
#include "helper_functions.h"
#include "hc_am.hpp"
#include "hcfftlib.h"

TEST(hcfft_1D_transform_test, func_correct_1D_transform_C2C ) {
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N1, HCFFT_C2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int hSize = N1;
  hcComplex* input = (hcComplex*)calloc(hSize, sizeof(hcComplex));
  hcComplex* output = (hcComplex*)calloc(hSize, sizeof(hcComplex));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < hSize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcComplex* idata = hc::am_alloc(hSize * sizeof(hcComplex), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcComplex) * hSize);
  hcComplex* odata = hc::am_alloc(hSize * sizeof(hcComplex), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcComplex) * hSize);
  status = hcfftExecC2C(*plan, idata, odata, HCFFT_FORWARD);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcComplex) * hSize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
   //FFTW work flow
  // input output arrays
  fftwf_complex *fftw_in,*fftw_out;
  int lengths[1] = {hSize};
  fftwf_plan p;
  fftw_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * hSize);
  fftw_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * hSize);
  // Populate inputs
  for(int i = 0; i < hSize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  // 1D forward plan
  p = fftwf_plan_many_dft( 1, lengths, 1, fftw_in, NULL, 1, 0, fftw_out, NULL, 1, 0, FFTW_FORWARD, FFTW_ESTIMATE);
  // Execute C2R
  fftwf_execute(p);
  // Check RMSE: If fails go for pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftwf_complex, hcComplex>(fftw_out, output, hSize)) {
    //Check Real Outputs
    for (int i =0; i < hSize; i++) {
      EXPECT_NEAR(fftw_out[i][0] , output[i].x, 0.1); 
    }
    //Check Imaginary Outputs
    for (int i =0; i < hSize; i++) {
      EXPECT_NEAR(fftw_out[i][1] , output[i].y, 0.1); 
    }
  }
  // Free up resources
  fftwf_destroy_plan(p);
  fftwf_free(fftw_in); fftwf_free(fftw_out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
