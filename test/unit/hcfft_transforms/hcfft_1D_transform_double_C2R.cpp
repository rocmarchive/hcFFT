#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "helper_functions.h"

TEST(hcfft_1D_transform_double_test, func_correct_1D_transform_Z2D ) {
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N1, HCFFT_Z2D);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Csize = (N1 / 2) + 1;
  int Rsize = N1;
  hcfftDoubleComplex* input = (hcfftDoubleComplex*)calloc(Csize, sizeof(hcfftDoubleComplex));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

  hcfftDoubleReal* output = (hcfftDoubleReal*)calloc(Rsize, sizeof(hcfftDoubleReal));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftDoubleComplex* idata = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftDoubleComplex) * Csize);
  hcfftDoubleReal* odata = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftDoubleReal) * Rsize);
  status = hcfftExecZ2D(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftDoubleReal) * Rsize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //FFTW work flow
  // input output arrays
  double *fftw_out; fftw_complex* fftw_in;
  int lengths[1] = {Rsize};
  fftw_plan p;
  fftw_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Csize);
  // Populate inputs
  for(int i = 0; i < Csize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  fftw_out = (double*) fftw_malloc(sizeof(double) * Rsize);
  // 1D forward plan
  p = fftw_plan_many_dft_c2r( 1, lengths, 1, fftw_in, NULL, 1, 0, fftw_out, NULL, 1, 0, FFTW_ESTIMATE | FFTW_HC2R);;
  // Execute C2R
  fftw_execute(p);

  // Check RMSE : IF fails go for 
  if (JudgeRMSEAccuracyReal<double, hcfftDoubleReal>(fftw_out, output, Rsize)) {
    //Check Real Outputs
    for (int i =0; i < Rsize; i++) {
      EXPECT_NEAR(fftw_out[i] , output[i], 1); 
    }
  }
  // Free up resources
  fftw_destroy_plan(p);
  fftw_free(fftw_in); fftw_free(fftw_out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
