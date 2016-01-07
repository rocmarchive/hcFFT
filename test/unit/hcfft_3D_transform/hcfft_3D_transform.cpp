#include "hcfft.h"
#include "gtest/gtest.h"
#include "clFFT.h"

#define VECTOR_SIZE 8
  
TEST(hcfft_3D_transform_test, func_correct_3D_transform_R2C ) {
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan3d(plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HCFFT_R2C);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = VECTOR_SIZE * VECTOR_SIZE * VECTOR_SIZE;
  int Csize = VECTOR_SIZE * VECTOR_SIZE * (1 + VECTOR_SIZE / 2);
  hcfftReal *input = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);
  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = rand();
  }
  hcfftComplex *output = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  Concurrency::array_view<hcfftReal> idata(Rsize, input);
  Concurrency::array_view<hcfftComplex> odata(Csize, output);
  status = hcfftExecR2C(*plan, &idata, &odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  odata.synchronize();
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  // clFFT work flow
  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufX, bufY;
  float *X, *Y;
  cl_event event = NULL;
  int ret = 0;
  size_t N1, N2, N3;
  N1 = N2 = N3 = VECTOR_SIZE;

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_3D;
  size_t clLengths[3] = { N1, N2, N3};

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  queue = clCreateCommandQueue( ctx, device, 0, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Setup clFFT. */
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetup(&fftSetup);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Allocate host & initialize data. */
  /* Only allocation shown for simplicity. */
  size_t realSize = N1 * N2 * N3;
  size_t complexSize = N3 * N2 * (1 + (N1 / 2)) * 2;

  X = (float *)calloc(realSize, sizeof(*X));
  Y = (float *)calloc(complexSize, sizeof(*Y));
  for(int i = 0; i < realSize; i++) {
          X[i] = input[i];
  }

  /* Prepare OpenCL memory objects and place data inside them. */
  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, realSize * sizeof(*X), NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  bufY = clCreateBuffer( ctx, CL_MEM_READ_WRITE, complexSize * sizeof(*Y), NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
  realSize * sizeof( *X ), X, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufY, CL_TRUE, 0,
  complexSize * sizeof( *Y ), Y, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  /*------------------------------------------------------R2C--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Execute the plan. */
  err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, &bufY, NULL);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Fetch results of calculations. */
  err = clEnqueueReadBuffer( queue, bufY, CL_TRUE, 0, complexSize * sizeof( *Y ), Y, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  //Compare the results of clFFT and HCFFT with 0.01 precision
  for(int i = 0; i < Csize; i++) {
    EXPECT_NEAR(odata[i].x, Y[2 * i], 0.01);
    EXPECT_NEAR(odata[i].y, Y[2 * i + 1], 0.01);
  }

  /* Release OpenCL memory objects. */
  clReleaseMemObject( bufX );
  clReleaseMemObject( bufY );

  free(X);
  free(Y);

  /* Release the plan. */
  err = clfftDestroyPlan( &planHandle );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Release clFFT library. */
  clfftTeardown( );

  /* Release OpenCL working objects. */
  clReleaseCommandQueue( queue );
  clReleaseContext( ctx );
}

TEST(hcfft_3D_transform_test, func_correct_3D_transform_C2R ) {
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan3d(plan, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, HCFFT_C2R);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Csize = VECTOR_SIZE * VECTOR_SIZE * (1 + VECTOR_SIZE / 2);
  int Rsize = VECTOR_SIZE * VECTOR_SIZE * VECTOR_SIZE;
  hcfftComplex *input = (hcfftComplex*)calloc(Csize, sizeof(hcfftComplex));
  hcfftReal *output = (hcfftReal*)calloc(Rsize, sizeof(hcfftReal));
  int seed = 123456789;
  srand(seed);
  // Populate the input
  for(int i = 0; i < Csize ; i++)
  {
    input[i].x = rand();
    input[i].y = rand();
  }
  Concurrency::array_view<hcfftComplex> idata(Csize, input);
  Concurrency::array_view<hcfftReal> odata(Rsize, output);
  status = hcfftExecC2R(*plan, &idata, &odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  odata.synchronize();
  status =  hcfftDestroy(*plan);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  // clFFT work flow
  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufX, bufY;
  float *X, *Y;
  cl_event event = NULL;
  int ret = 0;
  size_t N1, N2, N3;
  N1 = N2 = N3 = VECTOR_SIZE;

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_3D;
  size_t clLengths[3] = { N1, N2, N3};

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  queue = clCreateCommandQueue( ctx, device, 0, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Setup clFFT. */
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetup(&fftSetup);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Allocate host & initialize data. */
  /* Only allocation shown for simplicity. */
  size_t realSize = N1 * N2 * N3;
  size_t complexSize = N3 * N2 * (1 + (N1 / 2)) * 2;

  X = (float *)calloc(realSize, sizeof(*X));
  Y = (float *)calloc(complexSize, sizeof(*Y));
  for(int i = 0; i < complexSize / 2; i++) {
          Y[2 * i] = input[i].x;
          Y[2 * i + 1] = input[i].y;
  }

  /* Prepare OpenCL memory objects and place data inside them. */
  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, realSize * sizeof(*X), NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  bufY = clCreateBuffer( ctx, CL_MEM_READ_WRITE, complexSize * sizeof(*Y), NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
  realSize * sizeof( *X ), X, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufY, CL_TRUE, 0,
  complexSize * sizeof( *Y ), Y, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  /*------------------------------------------------------C2R--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Execute the plan. */
  err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &bufY, &bufX, NULL);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Fetch results of calculations. */
  err = clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, realSize * sizeof( *X ), X, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  EXPECT_EQ(err, CL_SUCCESS);

  //Compare the results of clFFT and HCFFT with 0.01 precision
  for(int i = 0; i < Rsize; i++) {
    EXPECT_NEAR(odata[i], X[i], 0.01);
  }

  /* Release OpenCL memory objects. */
  clReleaseMemObject( bufX );
  clReleaseMemObject( bufY );

  free(X);
  free(Y);

  /* Release the plan. */
  err = clfftDestroyPlan( &planHandle );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Release clFFT library. */
  clfftTeardown( );

  /* Release OpenCL working objects. */
  clReleaseCommandQueue( queue );
  clReleaseContext( ctx );
}
