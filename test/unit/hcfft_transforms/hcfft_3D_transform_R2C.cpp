#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "helper_functions.h"

TEST(hcfft_3D_transform_test, func_correct_3D_transform_R2C ) {
  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 2;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 2;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 2;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan3d(plan, N1, N2, N3, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N3 * N2 * N1;
  int Csize = N3 * N2 * (1 + N1 / 2);
  hcfftReal* input = (hcfftReal*)malloc(Rsize * sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = i%8;
  }

  hcfftComplex* output = (hcfftComplex*)malloc(Csize *sizeof(hcfftComplex));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftReal* idata = hc::am_alloc(Rsize * sizeof(hcfftReal), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftReal) * Rsize);
  hcfftComplex* odata = hc::am_alloc(Csize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftComplex) * Csize);
  status = hcfftExecR2C(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftComplex) * Csize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  //FFTW work flow
  // input output arrays
  float *in; fftwf_complex* out;
  fftwf_plan p;
  in = (float*) fftwf_malloc(sizeof(float) * Rsize);
  // Populate inputs
  for(int i = 0; i < Rsize ; i++) {
    in[i] = input[i];
  }
  out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Csize);
  // 3D forward plan
  p = fftwf_plan_dft_r2c_3d(N3, N2, N1, in, out, FFTW_ESTIMATE | FFTW_R2HC);;
  // Execute R2C
  fftwf_execute(p);
  // Check RMSE: If fails go for pointwise comparison
  if (JudgeRMSEAccuracyComplex<fftwf_complex, hcfftComplex>(out, output, Csize))
  { 
    //Check Real Outputs
    for (int i =0; i < Csize; i++) {
      EXPECT_NEAR(out[i][0] , output[i].x, 0.1); 
    }
    //Check Imaginary Outputs
    for (int i =0; i < Csize; i++) {
      EXPECT_NEAR(out[i][1] , output[i].y, 0.1); 
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

