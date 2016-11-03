#include "hcfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"

TEST(hcfft_1D_transform_double_test, func_correct_1D_transform_D2Z ) {
  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;
  // HCFFT work flow
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N1, HCFFT_D2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = N1;
  int Csize = (N1 / 2) + 1;
  hcfftDoubleReal* input = (hcfftDoubleReal*)calloc(Rsize, sizeof(hcfftDoubleReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = rand()%16;
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
  int lengths[1] = {Rsize};
  fftw_plan p;
  in = (double*) fftw_malloc(sizeof(double) * Rsize);
  // Populate inputs
  for(int i = 0; i < Rsize ; i++) {
    in[i] = input[i];
  }
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Csize);
  // 1D forward plan
  p = fftw_plan_many_dft_r2c( 1, lengths, 1, in, NULL, 1, 0, out, NULL, 1, 0, FFTW_ESTIMATE | FFTW_R2HC);;
  // Execute R2C
  fftw_execute(p);
  //Check Real Outputs
  for (int i =0; i < Csize; i++) {
    EXPECT_NEAR(out[i][0] , output[i].x, 0.01); 
  }
  //Check Imaginary Outputs
  for (int i = 0; i < Csize; i++) {
    EXPECT_NEAR(out[i][1] , output[i].y, 0.01); 
  }
  //Free up resources
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

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
  hcfftDoubleComplex* idata = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftDoubleComplex) * Csize);
  hcfftDoubleReal* odata = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftDoubleReal) * Rsize);
  status = hcfftExecZ2D(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  hc::am_copy(output, odata, sizeof(hcfftDoubleReal) * Rsize);
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
  //Check Real Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(fftw_out[i] , output[i], 1); 
  }
  // Free up resources
  fftw_destroy_plan(p);
  fftw_free(fftw_in); fftw_free(fftw_out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}

TEST(hcfft_1D_transform_double_test, func_correct_1D_transform_Z2Z ) {
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;
  hcfftHandle* plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, N1, HCFFT_Z2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int hSize = N1;
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
  int lengths[1] = {hSize};
  fftw_plan p;
  fftw_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hSize);
  fftw_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * hSize);
  // Populate inputs
  for(int i = 0; i < hSize ; i++) {
    fftw_in[i][0] = input[i].x;
    fftw_in[i][1] = input[i].y;
  }
  // 1D forward plan
  p = fftw_plan_many_dft( 1, lengths, 1, fftw_in, NULL, 1, 0, fftw_out, NULL, 1, 0, FFTW_FORWARD, FFTW_ESTIMATE);
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
  fftw_destroy_plan(p);
  fftw_free(fftw_in); fftw_free(fftw_out);
  free(input);
  free(output);
  hc::am_free(idata);
  hc::am_free(odata);
}
