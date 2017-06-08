#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "helper_functions.h"
#include "hc_am.hpp"
#include "hcfftlib.h"

TEST(hcfft_2D_transform_test, func_correct_2D_transform_Z2Z ) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle plan;
  hcfftResult status  = hcfftPlan2d(&plan, N1, N2, HCFFT_Z2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int hSize = N1 * N2;
  hcfftDoubleComplex* input = (hcfftDoubleComplex*)calloc(hSize, sizeof(hcfftDoubleComplex));
  hcfftDoubleComplex* output = (hcfftDoubleComplex*)calloc(hSize, sizeof(hcfftDoubleComplex));
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
  hcfftDoubleComplex* idata = hc::am_alloc(hSize * sizeof(hcfftDoubleComplex), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftDoubleComplex) * hSize);
  hcfftDoubleComplex* odata = hc::am_alloc(hSize * sizeof(hcfftDoubleComplex), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftDoubleComplex) * hSize);
  status = hcfftExecZ2Z(plan, idata, odata, HCFFT_FORWARD);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftDoubleComplex) * hSize);
  status =  hcfftDestroy(plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //FFTW work flow
  // input output arrays
  fftw_complex *fftw_in,*fftw_out;
  fftw_plan p;
  fftw_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hSize);
  fftw_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hSize);
  // Populate inputs
  for(int i = 0; i < hSize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  // 2D forward plan
  p = fftw_plan_dft_2d(N1, N2, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);
  // Execute C2R
  fftw_execute(p);

  // Check RMSE: If fails go for pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftw_complex, hcfftDoubleComplex>(fftw_out, output, hSize)) {
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
  fftw_destroy_plan(p);
  fftw_free(fftw_in); fftw_free(fftw_out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
