#include "hcfft.h"
#include "fftw3.h"
#include "../gtest/gtest.h"
#include "helper_functions.h"
#include "hc_am.hpp"
#include "hcfftlib.h"

TEST(hcfft_1D_transform_test, func_correct_1D_transform_R2C ) {
  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;
  // HCFFT work flow
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N1, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N1;
  int Csize = (N1 / 2) + 1;
  hcfftReal* input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = i%8;
  }

  hcComplex* output = (hcComplex*)calloc(Csize, sizeof(hcComplex));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftReal* idata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftReal) * Rsize);
  hcComplex* odata = hc::am_alloc(Csize * sizeof(hcComplex), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcComplex) * Csize);
  status = hcfftExecR2C(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcComplex) * Csize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  //FFTW work flow
  // input output arrays
  float *in; fftwf_complex* out;
  int lengths[1] = {Rsize};
  fftwf_plan p;
  in = (float*) fftwf_malloc(sizeof(float) * Rsize);
  // Populate inputs
  for(int i = 0; i < Rsize ; i++) {
    in[i] = input[i];
  }
  out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Csize);
  // 1D forward plan
  p = fftwf_plan_many_dft_r2c( 1, lengths, 1, in, NULL, 1, 0, out, NULL, 1, 0, FFTW_ESTIMATE | FFTW_R2HC);;
  // Execute R2C
  fftwf_execute(p);
  // Check RMSE: If fails go for pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftwf_complex, hcComplex>(out, output, Csize))
  { 
    //Check Real Outputs
    for (int i =0; i < Csize; i++) {
      EXPECT_NEAR(out[i][0] , output[i].x, 0.01); 
    }
    //Check Imaginary Outputs
    for (int i = 0; i < Csize; i++) {
      EXPECT_NEAR(out[i][1] , output[i].y, 0.01); 
    }
  }
  //Free up resources
  fftwf_destroy_plan(p);
  fftwf_free(in); fftwf_free(out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}


