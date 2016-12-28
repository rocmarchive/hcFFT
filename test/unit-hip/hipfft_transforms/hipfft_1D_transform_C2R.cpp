#include "hipfft.h"
#include "fftw3.h"
#include "../gtest/gtest.h"
#include "helper_functions.h"
#include "hip/hip_runtime_api.h"

TEST(hipfft_1D_transform_test, func_correct_1D_transform_C2R ) {
  size_t N1;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 1024;
  hipfftHandle plan;
  hipfftResult status  = hipfftPlan1d(&plan, N1, HIPFFT_C2R, 1);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  int Csize = (N1 / 2) + 1;
  int Rsize = N1;
  hipfftComplex* input = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Csize ; i++) {
    input[i].x = i%8;
    input[i].y = i%16;
  }

  hipfftReal* output = (hipfftReal*)calloc(Rsize, sizeof(hipfftReal));
  hipfftComplex* idata;
  hipfftReal* odata;
  hipMalloc(&idata, Csize * sizeof(hipfftComplex));
  hipMemcpy(idata, input, sizeof(hipfftComplex) * Csize, hipMemcpyHostToDevice);
  hipMalloc(&odata, Rsize * sizeof(hipfftReal));
  hipMemcpy(odata, output, sizeof(hipfftReal) * Rsize, hipMemcpyHostToDevice);
  status = hipfftExecC2R(plan, idata, odata);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(output, odata, sizeof(hipfftReal) * Rsize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
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
  if (JudgeRMSEAccuracyReal<float, hipfftReal>(fftw_out, output, Rsize)) {
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
  hipFree(idata);
  hipFree(odata);
}
