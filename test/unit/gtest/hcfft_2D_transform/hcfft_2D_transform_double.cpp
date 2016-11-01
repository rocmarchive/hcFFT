#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"

TEST(hcfft_2D_transform_test, func_correct_2D_transform_D2Z ) {
  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, N1, N2,  HCFFT_D2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N2 * N1;
  int Csize = N2 * (1 + N1 / 2);
  hcfftDoubleReal* input = (hcfftDoubleReal*)calloc(Rsize, sizeof(hcfftDoubleReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = i%8;
  }

  hcfftDoubleComplex* output = (hcfftDoubleComplex*)calloc(Csize, sizeof(hcfftDoubleComplex));
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hcfftDoubleReal* idata = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftDoubleReal) * Rsize);
  hcfftDoubleComplex* odata = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftDoubleComplex) * Csize);
  status = hcfftExecD2Z(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  hc::am_copy(output, odata, sizeof(hcfftDoubleComplex) * Csize);
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  //FFTW work flow
  // input output arrays
  double *in; fftw_complex* out;
  fftw_plan p;
  in = (double*) fftw_malloc(sizeof(double) * Rsize);
  // Populate inputs
  for(int i = 0; i < Rsize ; i++) {
    in[i] = input[i];
  }
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Csize);
  // 2D forward plan
  p = fftw_plan_dft_r2c_2d(N1, N2, in, out, FFTW_ESTIMATE | FFTW_R2HC);;
  // Execute R2C
  fftw_execute(p);
  //Check Real Outputs
  for (int i =0; i < Csize; i++) {
    EXPECT_NEAR(out[i][0] , output[i].x, 0.1); 
  }
  //Check Imaginary Outputs
  for (int i =0; i < Csize; i++) {
    EXPECT_NEAR(out[i][1] , output[i].y, 0.1); 
  }
  //Free up resources
  fftw_destroy_plan(p);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

TEST(hcfft_2D_transform_test, func_correct_2D_transform_Z2D ) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, N1, N2, HCFFT_Z2D);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Csize = N2 * (1 + N1 / 2);
  int Rsize = N2 * N1;
  hcfftDoubleComplex* input = (hcfftDoubleComplex*)calloc(Csize, sizeof(hcfftDoubleComplex));
  hcfftDoubleReal* output = (hcfftDoubleReal*)calloc(Rsize, sizeof(hcfftDoubleReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");
  hcfftDoubleComplex* idata = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftDoubleComplex) * Csize);
  hcfftDoubleReal* odata = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  hc::am_copy(odata, output, sizeof(hcfftDoubleReal) * Rsize);
  status = hcfftExecZ2D(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  hc::am_copy(output, odata, sizeof(hcfftDoubleReal) * Rsize);
  status =  hcfftDestroy(*plan);
  //FFTW work flow
  // input output arrays
  double *fftw_out; fftw_complex* fftw_in;
  fftw_plan p;
  fftw_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Csize);
  // Populate inputs
  for(int i = 0; i < Csize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  fftw_out = (double*) fftw_malloc(sizeof(double) * Rsize);
  // 2D forward plan
  p = fftw_plan_dft_c2r_2d(N1, N2, fftw_in, fftw_out, FFTW_ESTIMATE | FFTW_HC2R);;
  // Execute C2R
  fftw_execute(p);
  //Check Real Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(fftw_out[i] , output[i], 0.1); 
  }
  // Free up resources
  fftw_free(fftw_in); fftw_free(fftw_out); 
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

TEST(hcfft_2D_transform_test, func_correct_2D_transform_Z2Z ) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan2d(plan, N1, N2, HCFFT_Z2Z);
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
  hcfftDoubleComplex* idata = hc::am_alloc(hSize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftDoubleComplex) * hSize);
  hcfftDoubleComplex* odata = hc::am_alloc(hSize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftDoubleComplex) * hSize);
  status = hcfftExecZ2Z(*plan, idata, odata, HCFFT_FORWARD);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  hc::am_copy(output, odata, sizeof(hcfftDoubleComplex) * hSize);
  status =  hcfftDestroy(*plan);
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
  //Check Real Outputs
  for (int i =0; i < hSize; i++) {
    EXPECT_NEAR(fftw_out[i][0] , output[i].x, 0.1); 
  }
  //Check Imaginary Outputs
  for (int i =0; i < hSize; i++) {
    EXPECT_NEAR(fftw_out[i][1] , output[i].y, 0.1); 
  }
  // Free up resources
  fftw_free(fftw_in); fftw_free(fftw_out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
