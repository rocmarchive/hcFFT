#include "hipfft.h"
#include "fftw3.h"
#include "../gtest/gtest.h"
#include "helper_functions.h"
#include "hip/hip_runtime.h"

TEST(hipfft_1D_transform_test, func_correct_1D_transform_R2C ) {
//  putenv((char*)"GTEST_BREAK_ON_FAILURE=0");
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;

  // HIPFFT work flow
  hipfftHandle plan;
  hipfftResult status  = hipfftPlan1d(&plan, N1, HIPFFT_R2C, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  int Rsize = N1;
  int Csize = (N1 / 2) + 1;
  hipfftReal* input = (hipfftReal*)calloc(Rsize, sizeof(hipfftReal));
  hipfftComplex* output = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = i%8;
  }

  hipfftReal* idata;
  hipfftComplex* odata;
  hipMalloc(&idata, Rsize * sizeof(hipfftReal));
  hipMemcpy(idata, input, sizeof(hipfftReal) * Rsize, hipMemcpyHostToDevice);
  hipMalloc(&odata, Csize * sizeof(hipfftComplex));
  hipMemcpy(odata, output, sizeof(hipfftComplex) * Csize, hipMemcpyHostToDevice);
  status = hipfftExecR2C(plan, idata, odata);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(output, odata, sizeof(hipfftComplex) * Csize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

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
  if (JudgeRMSEAccuracyComplex<fftwf_complex, hipfftComplex>(out, output, Csize))
  { 
    //Check Real Outputs
    for (int i =0; i < Csize; i++) {
      ASSERT_NEAR(out[i][0] , output[i].x, 0.01); 
    }
    //Check Imaginary Outputs
    for (int i = 0; i < Csize; i++) {
      ASSERT_NEAR(out[i][1] , output[i].y, 0.01); 
    }
  }

  //Free up resources
  fftwf_destroy_plan(p);
  fftwf_free(in); fftwf_free(out);
  free(input);
  free(output);
  hipFree(idata);
  hipFree(odata);
}


