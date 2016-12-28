#include "hipfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "hip/hip_runtime_api.h"

TEST(hipfft_3D_transform_test, func_correct_3D_transform_C2R_RTT) {
  size_t N1, N2, N3;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 2;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 2;
  N3 = my_argc > 3 ? atoi(my_argv[3]) : 2;
  hipfftHandle plan;
  hipfftResult status  = hipfftPlan3d(&plan, N1, N2, N3, HIPFFT_R2C);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  int Rsize = N3 * N2 * N1;
  int Csize = N3 * N2 * (1 + N1 / 2);
  hipfftReal* inputR2C = (hipfftReal*)malloc(Rsize * sizeof(hipfftReal));
  int seed = 123456789;
  srand(seed);

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    inputR2C[i] = i%8;
  }

  hipfftComplex* outputR2C = (hipfftComplex*)malloc(Csize *sizeof(hipfftComplex));
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
  status  = hipfftPlan3d(&plan, N1, N2, N3, HIPFFT_C2R);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipfftComplex* inputC2R = (hipfftComplex*)malloc(Csize * sizeof(hipfftComplex));
  hipfftReal* outputC2R = (hipfftReal*)malloc(Rsize * sizeof(hipfftReal));

  // Populate the input
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
  //Check Real Inputs and  Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(inputR2C[i] , outputC2R[i]/Rsize, 0.1); 
  }
  // Free up resources
  free(inputC2R);
  free(outputC2R);
  free(inputR2C);
  free(outputR2C);
  hipFree(devIpC2R);
  hipFree(devOpC2R);
  hipFree(devIpR2C);
  hipFree(devOpR2C);
}

