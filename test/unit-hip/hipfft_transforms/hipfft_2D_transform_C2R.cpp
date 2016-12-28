#include "hipfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "hip/hip_runtime_api.h"

TEST(hipfft_2D_transform_test, func_correct_2D_transform_C2R_RTT) {
  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;
  hipfftHandle plan;
  hipfftResult status  = hipfftPlan2d(&plan, N1, N2,  HIPFFT_R2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  int Rsize = N1 * N2;
  int Csize = N2 * (1 + N1 / 2);
  hipfftReal* inputR2C = (hipfftReal*)calloc(Rsize, sizeof(hipfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    inputR2C[i] = i%8;
  }

  hipfftComplex* outputR2C = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));
  hipfftReal* devIpR2C; 
  hipfftComplex* devOpR2C;
  hipMalloc(&devIpR2C, Rsize * sizeof(hipfftReal));
  hipMemcpy(devIpR2C, inputR2C, sizeof(hipfftReal) * Rsize, hipMemcpyHostToDevice);
  hipMalloc(&devOpR2C, Csize * sizeof(hipfftComplex));
  hipMemcpy(devOpR2C, outputR2C, sizeof(hipfftComplex) * Csize, hipMemcpyHostToDevice);
  status = hipfftExecR2C(plan, devIpR2C, devOpR2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(outputR2C, devOpR2C, sizeof(hipfftComplex) * Csize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  //plan = NULL;
  status  = hipfftPlan2d(&plan, N1, N2, HIPFFT_C2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipfftComplex* inputC2R = (hipfftComplex*)calloc(Csize, sizeof(hipfftComplex));
  hipfftReal* outputC2R = (hipfftReal*)calloc(Rsize, sizeof(hipfftReal));

  // Populate the input of C2R with output ofo R2C
  for(int i = 0; i < Csize ; i++) {
    inputC2R[i].x = outputR2C[i].x;
    inputC2R[i].y = outputR2C[i].y;
  }

  hipfftComplex* devIpC2R; 
  hipfftReal* devOpC2R;
  hipMalloc(&devIpC2R, Csize * sizeof(hipfftComplex));
  hipMemcpy(devIpC2R, inputC2R, sizeof(hipfftComplex) * Csize, hipMemcpyHostToDevice); 
  hipMalloc(&devOpC2R, Rsize * sizeof(hipfftReal));
  hipMemcpy(devOpC2R, outputC2R, sizeof(hipfftReal) * Rsize, hipMemcpyHostToDevice);
  status = hipfftExecC2R(plan, devIpC2R, devOpC2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(outputC2R, devOpC2R, sizeof(hipfftReal) * Rsize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  //Check Real Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(outputC2R[i]/(Rsize), inputR2C[i], 1); 
  }
  // Free up resources
  free(inputR2C);
  free(outputR2C);
  free(inputC2R);
  free(outputC2R);
  hipFree(devOpC2R);
  hipFree(devOpR2C);
  hipFree(devIpC2R);
  hipFree(devIpR2C);
}

