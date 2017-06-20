#include "hipfft.h"
#include "../gtest/gtest.h"
#include "fftw3.h"
#include "hip/hip_runtime.h"
#include <iostream>
using namespace std;

TEST(hipfft_2D_transform_test, func_correct_2D_transform_Z2D_RTT) {

  size_t N1, N2;
  N1 = my_argc > 1 ? atoi(my_argv[1]) : 8;
  N2 = my_argc > 2 ? atoi(my_argv[2]) : 8;

  // HIPFFT work flow  
  hipfftHandle plan;
  // First Real to Complex tranformation
  hipfftResult status  = hipfftPlan2d(&plan, N1, N2,  HIPFFT_D2Z);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  int Rsize = N2 * N1;
  int Csize = N1 * (1 + N2 / 2);
  hipfftDoubleReal* inputD2Z = (hipfftDoubleReal*)calloc(Rsize, sizeof(hipfftDoubleReal));
  hipfftDoubleComplex* outputD2Z = (hipfftDoubleComplex*)calloc(Csize, sizeof(hipfftDoubleComplex));

  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    inputD2Z[i] = i%8;
  }

  hipfftDoubleReal* devIpD2Z;
  hipfftDoubleComplex* devOpD2Z;
  hipMalloc(&devIpD2Z, Rsize * sizeof(hipfftDoubleReal));
  hipMemcpy(devIpD2Z, inputD2Z, sizeof(hipfftDoubleReal) * Rsize, hipMemcpyHostToDevice); 
  hipMalloc(&devOpD2Z, Csize * sizeof(hipfftDoubleComplex));
  hipMemcpy(devOpD2Z, outputD2Z, sizeof(hipfftDoubleComplex) * Csize, hipMemcpyHostToDevice);
  status = hipfftExecD2Z(plan, devIpD2Z, devOpD2Z);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(outputD2Z, devOpD2Z, sizeof(hipfftDoubleComplex) * Csize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  // Next Complex to Real transformation
  status  = hipfftPlan2d(&plan, N1, N2, HIPFFT_Z2D);
  EXPECT_EQ(status, HIPFFT_SUCCESS);

  hipfftDoubleComplex* inputZ2D = (hipfftDoubleComplex*)calloc(Csize, sizeof(hipfftDoubleComplex));
  hipfftDoubleReal* outputZ2D = (hipfftDoubleReal*)calloc(Rsize, sizeof(hipfftDoubleReal));

  // Populate the input to Z2D with output of D2Z
  for(int i = 0; i < Csize ; i++) {
    inputZ2D[i].x = outputD2Z[i].x;
    inputZ2D[i].y = outputD2Z[i].y;
  }

  hipfftDoubleComplex* devIpZ2D; 
  hipfftDoubleReal* devOpZ2D;
  hipMalloc(&devIpZ2D, Csize * sizeof(hipfftDoubleComplex));
  hipMemcpy(devIpZ2D, inputZ2D, sizeof(hipfftDoubleComplex) * Csize, hipMemcpyHostToDevice);
  hipMalloc(&devOpZ2D, Rsize * sizeof(hipfftDoubleReal));
  hipMemcpy(devOpZ2D, outputZ2D, sizeof(hipfftDoubleReal) * Rsize, hipMemcpyHostToDevice);
  status = hipfftExecZ2D(plan, devIpZ2D, devOpZ2D);
  EXPECT_EQ(status, HIPFFT_SUCCESS);
  hipMemcpy(outputZ2D, devOpZ2D, sizeof(hipfftDoubleReal) * Rsize, hipMemcpyDeviceToHost);
  status =  hipfftDestroy(plan);
  
  //Check Real Inputs and Outputs
  for (int i =0; i < Rsize; i++) {
    EXPECT_NEAR(inputD2Z[i] , outputZ2D[i]/Rsize, 0.1); 
  }

  // Free up resources
  free(inputD2Z);
  free(outputD2Z);
  free(inputZ2D);
  free(outputZ2D);
  hipFree(devIpZ2D);
  hipFree(devOpZ2D);
  hipFree(devIpD2Z);
  hipFree(devOpD2Z);
}
