#include "hcfft.h"
#include "../gtest/gtest.h"
#include "clFFT.h"

#define VECTOR_SIZE 1024

TEST(hcfft_1D_transform_double_test, func_correct_1D_transform_D2Z ) {
  // HCFFT work flow
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, VECTOR_SIZE, HCFFT_D2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Rsize = VECTOR_SIZE;
  int Csize = (VECTOR_SIZE / 2) + 1;
  hcfftDoubleReal *input = (hcfftDoubleReal*)calloc(Rsize, sizeof(hcfftDoubleReal));
  int seed = 123456789;
  srand(seed);
  // Populate the input
  for(int i = 0; i < Rsize ; i++) {
    input[i] = rand();
  }
  hcfftDoubleComplex *output = (hcfftDoubleComplex*)calloc(Csize, sizeof(hcfftDoubleComplex));

  std::vector<accelerator> accs = accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");

  hcfftDoubleReal *idata = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftDoubleReal) * Rsize);

  hcfftDoubleComplex *odata = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftDoubleComplex) * Csize);

  status = hcfftExecD2Z(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  hc::am_copy(output, odata, sizeof(hcfftDoubleComplex) * Csize);

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
  double *X, *Y;
  cl_event event = NULL;
  int ret = 0;
  size_t N1;
  N1 = VECTOR_SIZE;

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_1D;
  size_t clLengths[1] = { N1};
  size_t ipStrides[1] = { 1 };
  size_t ipDistance = N1;
  size_t opStrides[1] = {1};
  size_t opDistance = 1 + N1/2;

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );
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
  size_t realSize = N1;
  size_t complexSize = (1 + (N1 / 2)) * 2;

  X = (double *)calloc(realSize, sizeof(*X));
  Y = (double *)calloc(complexSize, sizeof(*Y));
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

  /*------------------------------------------------------D2Z--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_DOUBLE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanInStride(planHandle, dim, ipStrides );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanOutStride(planHandle, dim, opStrides );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanDistance(planHandle, ipDistance, opDistance );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
  EXPECT_EQ(err, CL_SUCCESS);
 
  /* Execute the plan. */
  err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, &bufY, NULL);
  EXPECT_EQ(err, CL_SUCCESS);
 
  /* Fetch results of calculations. */
  err = clEnqueueReadBuffer( queue, bufY, CL_TRUE, 0, complexSize * sizeof( *Y ), Y, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  EXPECT_EQ(err, CL_SUCCESS);

  //Compare the results of clFFT and HCFFT with 0.01 precision
  for(int i = 0; i < Csize; i++) {
    EXPECT_NEAR(output[i].x, Y[2 * i], 0.01);
    EXPECT_NEAR(output[i].y, Y[2 * i + 1], 0.01);
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

  free(input);
  free(output);

  hc::am_free(idata);
  hc::am_free(odata);
}

TEST(hcfft_1D_transform_double_test, func_correct_1D_transform_Z2D ) {
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, VECTOR_SIZE, HCFFT_Z2D);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int Csize = (VECTOR_SIZE / 2) + 1;
  int Rsize = VECTOR_SIZE;
  hcfftDoubleComplex *input = (hcfftDoubleComplex*)calloc(Csize, sizeof(hcfftDoubleComplex));
  int seed = 123456789;
  srand(seed);
  // Populate the input
  for(int i = 0; i < Csize ; i++)
  {
    input[i].x = rand();
    input[i].y = rand();
  }
  hcfftDoubleReal *output = (hcfftDoubleReal*)calloc(Rsize, sizeof(hcfftDoubleReal));

  std::vector<accelerator> accs = accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");

  hcfftDoubleComplex *idata = hc::am_alloc(Csize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftDoubleComplex) * Csize);

  hcfftDoubleReal *odata = hc::am_alloc(Rsize * sizeof(hcfftDoubleReal), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftDoubleReal) * Rsize);

  status = hcfftExecZ2D(*plan, idata, odata);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  hc::am_copy(output, odata, sizeof(hcfftDoubleReal) * Rsize);

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
  double *X, *Y;
  cl_event event = NULL;
  int ret = 0;
  size_t N1;
  N1 = VECTOR_SIZE;

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_1D;
  size_t clLengths[1] = { N1};
  size_t ipStrides[1] = { 1 };
  size_t ipDistance = 1 + N1/2;
  size_t opStrides[1] = {1};
  size_t opDistance = N1;
  cl_float scale = 1.0;

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );
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
  size_t realSize = N1;
  size_t complexSize = (1+(N1/2)) * 2;

  X = (double *)calloc(realSize, sizeof(*X));
  Y = (double *)calloc(complexSize, sizeof(*Y));
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

  /*------------------------------------------------------Z2D--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_DOUBLE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanInStride(planHandle, dim, ipStrides );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanOutStride(planHandle, dim, opStrides );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanDistance(planHandle, ipDistance, opDistance );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanScale(planHandle, CLFFT_BACKWARD, scale );
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
    EXPECT_NEAR(output[i], X[i], 0.01);
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

  free(input);
  free(output);

  hc::am_free(idata);
  hc::am_free(odata);
}

TEST(hcfft_1D_transform_double_test, func_correct_1D_transform_Z2Z ) {
  hcfftHandle *plan = NULL;
  hcfftResult status  = hcfftPlan1d(plan, VECTOR_SIZE, HCFFT_Z2Z);
  EXPECT_EQ(status, HCFFT_SUCCESS);
  int hSize = VECTOR_SIZE;
  hcfftDoubleComplex *input = (hcfftDoubleComplex*)calloc(hSize, sizeof(hcfftDoubleComplex));
  hcfftDoubleComplex *output = (hcfftDoubleComplex*)calloc(hSize, sizeof(hcfftDoubleComplex));
  int seed = 123456789;
  srand(seed);
  // Populate the input
  for(int i = 0; i < hSize ; i++)
  {
    input[i].x = rand();
    input[i].y = rand();
  }

  std::vector<accelerator> accs = accelerator::get_all();
  assert(accs.size() && "Number of Accelerators == 0!");

  hcfftDoubleComplex *idata = hc::am_alloc(hSize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(idata, input, sizeof(hcfftDoubleComplex) * hSize);

  hcfftDoubleComplex *odata = hc::am_alloc(hSize * sizeof(hcfftDoubleComplex), accs[1], 0);
  hc::am_copy(odata,  output, sizeof(hcfftDoubleComplex) * hSize);

  status = hcfftExecZ2Z(*plan, idata, odata, HCFFT_FORWARD);
  EXPECT_EQ(status, HCFFT_SUCCESS);

  hc::am_copy(output, odata, sizeof(hcfftDoubleComplex) * hSize);

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
  double *X, *Y;
  cl_event event = NULL;
  int ret = 0;
  size_t N1;
  N1 = VECTOR_SIZE;

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_1D;
  size_t clLengths[1] = { N1};
  size_t ipStrides[1] = {1};
  size_t ipDistance = N1;
  size_t opStrides[1] = {1};
  size_t  opDistance = N1;

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );
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
  size_t size = N1 * 2;

  X = (double *)calloc(size, sizeof(*X));
  Y = (double *)calloc(size, sizeof(*Y));
  for(int i = 0; i < size / 2; i++) {
          X[2 * i] = input[i].x;
          X[2 * i + 1] = input[i].y;
  }

  /* Prepare OpenCL memory objects and place data inside them. */
  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, size * sizeof(*X), NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  bufY = clCreateBuffer( ctx, CL_MEM_READ_WRITE, size * sizeof(*Y), NULL, &err );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
  size * sizeof( *X ), X, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufY, CL_TRUE, 0,
  size * sizeof( *Y ), Y, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  /*------------------------------------------------------Z2Z--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_DOUBLE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanInStride(planHandle, dim, ipStrides );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanOutStride(planHandle, dim, opStrides );
  EXPECT_EQ(err, CL_SUCCESS);

  err = clfftSetPlanDistance(planHandle, ipDistance, opDistance );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Execute the plan. */
  err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, &bufY, NULL);
  EXPECT_EQ(err, CL_SUCCESS);

  /* Fetch results of calculations. */
  err = clEnqueueReadBuffer( queue, bufY, CL_TRUE, 0, size * sizeof( *Y ), Y, 0, NULL, NULL );
  EXPECT_EQ(err, CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  EXPECT_EQ(err, CL_SUCCESS);

  //Compare the results of clFFT and HCFFT with 0.01 precision
  for(int i = 0; i < hSize; i++) {
    EXPECT_NEAR(output[i].x, Y[2 * i], 0.01);
    EXPECT_NEAR(output[i].y, Y[2 * i + 1], 0.01);
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

  free(input);
  free(output);

  hc::am_free(idata);
  hc::am_free(odata);
}
