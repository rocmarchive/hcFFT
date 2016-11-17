#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"

TEST(hcfft_2D_transform_test, func_correct_2D_transform_R2C ) {
  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, N1, N2,  HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N1 * N2;
  int Csize = N2 * (1 + N1 / 2);
  hcfftReal* input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = i%8;
  }

  hcfftComplex* output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
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
  // 2D forward plan
  p = fftwf_plan_dft_r2c_2d(N2, N1, in, out, FFTW_ESTIMATE | FFTW_R2HC);;
  // Execute R2C
  fftwf_execute(p);
  //Check Real Outputs
  for (int i =0; i < Csize; i++) {
    EXPECT_NEAR(out[i][0] , output[i].x, 0.1); 
  }
  //Check Imaginary Outputs
  for (int i =0; i < Csize; i++) {
    EXPECT_NEAR(out[i][1] , output[i].y, 0.1); 
  }
  //Free up resources
  fftwf_destroy_plan(p);
  fftwf_free(in); fftwf_free(out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

TEST(hcfft_2D_transform_test, func_correct_2D_transform_C2R ) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, N1, N2, HCFFT_C2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Csize = N2 * (1 + N1 / 2);
  int Rsize = N2 * N1;
  hcfftComplex* input = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  hcfftReal* output = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

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
  fftwf_plan p;
  fftw_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Csize);
  // Populate inputs
  for(int i = 0; i < Csize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  fftw_out = (float*) fftwf_malloc(sizeof(float) * Rsize);
  // 2D forward plan
  p = fftwf_plan_dft_c2r_2d(N2, N1, fftw_in, fftw_out, FFTW_ESTIMATE | FFTW_HC2R);;
  // Execute C2R
  fftwf_execute(p);
  //Check Real Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(fftw_out[i] , output[i], 0.01); 
  }
  // Free up resources
  fftwf_destroy_plan(p);
  fftwf_free(fftw_in); fftwf_free(fftw_out); 
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

TEST(hcfft_2D_transform_test, func_correct_2D_transform_C2C ) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, N1, N2, HCFFT_C2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int hSize = N1 * N2;
  hcfftComplex* input = (hcfftComplex*)calloc(hSize, sizeof(hcfftComplex));
  hcfftComplex* output = (hcfftComplex*)calloc(hSize, sizeof(hcfftComplex));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < hSize ; i++) {
    input[i].x = i%8;
    input[i].y = i%8;
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hc::accelerator_view accl_view = accs[1].get_default_view();
  hcfftComplex* idata = hc::am_alloc(hSize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(input, idata, sizeof(hcfftComplex) * hSize);
  hcfftComplex* odata = hc::am_alloc(hSize * sizeof(hcfftComplex), accs[1], 0);
  accl_view.copy(output, odata, sizeof(hcfftComplex) * hSize);
  status = hcfftExecC2C(*plan, idata, odata, HCFFT_FORWARD);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  accl_view.copy(odata, output, sizeof(hcfftComplex) * hSize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //FFTW work flow
  // input output arrays
  fftwf_complex *fftw_in,*fftw_out;
  fftwf_plan p;
  fftw_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * hSize);
  fftw_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * hSize);
  // Populate inputs
  for(int i = 0; i < hSize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  // 2D forward plan
  p = fftwf_plan_dft_2d(N2, N1, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);
  // Execute C2R
  fftwf_execute(p);
  //Check Real Outputs
  for (int i =0; i < hSize; i++) {
    EXPECT_NEAR(fftw_out[i][0] , output[i].x, 0.1); 
  }
  //Check Imaginary Outputs
  for (int i =0; i < hSize; i++) {
    EXPECT_NEAR(fftw_out[i][1] , output[i].y, 0.1); 
  }
  // Free up resources
  fftwf_destroy_plan(p);
  fftwf_free(fftw_in); fftwf_free(fftw_out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
