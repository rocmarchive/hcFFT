#include "hcfft.h"
#include "fftw3.h"
#include "../gtest/gtest.h"
#include "helper_functions.h"

TEST(hcfft_1D_transform_test, func_correct_1D_transform_C2R ) {
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N1, HCFFT_C2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Csize = (N1 / 2) + 1;
  int Rsize = N1;
  hcfftComplex* input = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

  hcfftReal* output = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftComplex* idata = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftComplex) * Csize);
  hcfftReal* odata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftReal) * Rsize);
  status = hcfftExecC2R(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftReal) * Rsize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //FFTW work flow
  // input output arrays
  float *fftw_out; fftwf_complex* fftw_in;
  int lengths[1] = {Rsize};
  fftwf_plan p;
  fftw_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Csize);
  // Populate inputs
  for(int i = 0; i < Csize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  fftw_out = (float*) fftwf_malloc(sizeof(float) * Rsize);
  // 1D forward plan
  p = fftwf_plan_many_dft_c2r( 1, lengths, 1, fftw_in, NULL, 1, 0, fftw_out, NULL, 1, 0, FFTW_ESTIMATE | FFTW_HC2R);;
  // Execute C2R
  fftwf_execute(p);

  // Check RMSE : IF fails go for 
  if (JudgeRMSEAccuracyReal<float, hcfftReal>(fftw_out, output, Rsize)) {
    //Check Real Outputs
    for (int i =0; i < Rsize; i++) {
      EXPECT_NEAR(fftw_out[i] , output[i], 0.1); 
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
