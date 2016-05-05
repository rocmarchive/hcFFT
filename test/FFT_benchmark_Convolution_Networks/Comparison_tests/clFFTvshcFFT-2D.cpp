#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <hcfftlib.h>
#include <dlfcn.h>
#include <map>
#include <hc.hpp>
#include <cfloat>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <vector>
#include "clFFT.h"
#include <thread>

#define COUNT 100

template <typename T>
  T average(const std::vector<std::chrono::duration<T>> &data) {
  T avg_duration = 0;
  for(auto &i : data)
    avg_duration += i.count();
  return avg_duration/data.size();
}

std::chrono::time_point<std::chrono::high_resolution_clock> starttimer, endtimer;
std::vector<std::chrono::duration<double>> elapsed_pfe;  

void hcfft_2d_r2c(size_t N1, size_t N2)
{
  FFTPlan plan;
  const hcfftDim dimension = HCFFT_2D;
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftResLocation location = HCFFT_OUTOFPLACE;
  hcfftResTransposed transposeType = HCFFT_NOTRANSPOSE;
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t *ipStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t *opStrides = (size_t*)malloc(sizeof(size_t) * dimension);

  hcfftPlanHandle planhandle;
  hcfftPrecision precision = HCFFT_SINGLE;
  length[0] = N1;
  length[1] = N2;

  ipStrides[0] = 1;
  ipStrides[1] = length[0];

  opStrides[0] = 1;
  opStrides[1] = 1 + length[0]/2;

  size_t ipDistance = length[1] * length[0];
  size_t opDistance = length[1] * (1 + length[0]/2);

  int realsize, cmplexsize;
  realsize = length[0] * length[1];
  cmplexsize = length[1] * (1 + (length[0] / 2)) * 2;

  std::vector<accelerator> accs = accelerator::get_all();

  // Initialize host variables ----------------------------------------------
  float* ipHost = (float*)calloc(realsize, sizeof(float));
  float* opHost = (float*)calloc(cmplexsize, sizeof(float));

  for(int  i = 0; i < N2 ; i++) {
    for(int  j = 0; j < N1 ; j++) {
      ipHost[i * N1 + j] = i * N1 + j + 1;
    }
  }

  float* ipDev = (float*)am_alloc(realsize * sizeof(float), accs[1], 0);
  float* opDev = (float*)am_alloc(cmplexsize * sizeof(float), accs[1], 0);

  // Copy input contents to device from host
  hc::am_copy(ipDev, ipHost, realsize * sizeof(float));
  hc::am_copy(opDev, opHost, cmplexsize * sizeof(float));

  hcfftLibType libtype = HCFFT_R2CD2Z;

  hcfftStatus status = plan.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir, precision, libtype);
  if(status != HCFFT_SUCCEEDS) {
    cout << " Create plan error " << endl;
  }

  status = plan.hcfftSetAcclView(planhandle, accs[1].create_view());
  if(status != HCFFT_SUCCEEDS) {
    cout << " set accleration view error " << endl;
  }

  status = plan.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set plan error " << endl;
  }

  status = plan.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set result error " << endl;
  }

  status = plan.hcfftSetPlanInStride(planhandle, dimension, ipStrides );

  if(status != HCFFT_SUCCEEDS) {
    cout<<" hcfftSetPlanInStride error "<<endl;
  }

  status = plan.hcfftSetPlanOutStride(planhandle, dimension, opStrides );

  if(status != HCFFT_SUCCEEDS) {
    cout<<"hcfftSetPlanOutStride error "<<endl;
  }

  status = plan.hcfftSetPlanDistance(planhandle, ipDistance, opDistance );

  if(status != HCFFT_SUCCEEDS) {
    cout<<"hcfftSetPlanDistance error "<<endl;
  }

  /*---------------------R2C--------------------------------------*/
  status = plan.hcfftSetLayout(planhandle, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set layout error " << endl;
  }

  status = plan.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCEEDS) {
    cout << " bake plan error " << endl;
  }

  elapsed_pfe.clear();
  accs[1].get_default_view().wait();

  for(int i = 0; i < COUNT; i++)
  {
  starttimer = std::chrono::high_resolution_clock::now();
  status = plan.hcfftEnqueueTransform(planhandle, dir, ipDev, opDev, NULL);
  if(status != HCFFT_SUCCEEDS) {
    cout << " Transform error " << endl;
  }
  accs[1].get_default_view().wait();
  endtimer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> dur = endtimer - starttimer;
  if(i != 0)
    elapsed_pfe.push_back(dur);
  }

  double Avg_time = average(elapsed_pfe);
  double time_in_ms = Avg_time * 1e3;
  cout << "HCFFT Kernel execution time R2C Transform for size "<< N1 << "x" << N2 << " in <ms>:" << time_in_ms <<endl;

  // Copy Device output  contents back to host
  hc::am_copy(opHost, opDev, cmplexsize * sizeof(float));

  status = plan.hcfftDestroyPlan(&planhandle);
  if(status != HCFFT_SUCCEEDS) {
    cout << " destroy plan error " << endl;
  }

  hc::am_free(ipDev);
  hc::am_free(opDev);

  free(ipHost);
  free(opHost);

  free(length);
  free(ipStrides);
  free(opStrides);

}

void clfft_2d_r2c(size_t N1, size_t N2)
{
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

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_2D;
  size_t clLengths[2] = { N1, N2};
  size_t ipStrides[2] = {1, N1};
  size_t ipDistance = N2 * N1;
  size_t opStrides[2] = {1, 1 + N1/2};
  size_t opDistance = N2*(1 + N1/2);

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  assert(err == CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );
  assert(err == CL_SUCCESS);

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  assert(err ==  CL_SUCCESS);

  queue = clCreateCommandQueue( ctx, device, 0, &err );
  assert(err ==  CL_SUCCESS);

  /* Setup clFFT. */
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  assert(err ==  CL_SUCCESS);

  err = clfftSetup(&fftSetup);
  assert(err ==  CL_SUCCESS);

  /* Allocate host & initialize data. */
  /* Only allocation shown for simplicity. */
  size_t realSize = N1 * N2;
  size_t complexSize = N2 * (1 + (N1 / 2)) * 2;

  X = (float *)calloc(realSize, sizeof(*X));
  Y = (float *)calloc(complexSize, sizeof(*Y));
  for(int i = 0; i < realSize; i++) {
          X[i] = i + 1;
  }

  /* Prepare OpenCL memory objects and place data inside them. */
  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, realSize * sizeof(*X), NULL, &err );
  assert(err ==  CL_SUCCESS);

  bufY = clCreateBuffer( ctx, CL_MEM_READ_WRITE, complexSize * sizeof(*Y), NULL, &err );
  assert(err ==  CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
  realSize * sizeof( *X ), X, 0, NULL, NULL );
  assert(err ==  CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufY, CL_TRUE, 0,
  complexSize * sizeof( *Y ), Y, 0, NULL, NULL );
  assert(err ==  CL_SUCCESS);

  /*------------------------------------------------------R2C--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  assert(err ==  CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  assert(err ==  CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  assert(err ==  CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
  assert(err ==  CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  assert(err ==  CL_SUCCESS);

  err = clfftSetPlanInStride(planHandle, dim, ipStrides );
  assert(err ==  CL_SUCCESS);

  err = clfftSetPlanOutStride(planHandle, dim, opStrides );
  assert(err ==  CL_SUCCESS);

  err = clfftSetPlanDistance(planHandle, ipDistance, opDistance );
  assert(err ==  CL_SUCCESS);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
  assert(err ==  CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  assert(err ==  CL_SUCCESS);

  elapsed_pfe.clear();
  for(int i = 0; i < COUNT; i++)
  {
  starttimer = std::chrono::high_resolution_clock::now();

  /* Execute the plan. */
  err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, &bufY, NULL);
  assert(err ==  CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  assert(err ==  CL_SUCCESS);

  endtimer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> dur = endtimer - starttimer;
  if(i != 0)
    elapsed_pfe.push_back(dur);
  }

  double Avg_time = average(elapsed_pfe);
  double time_in_ms = Avg_time * 1e3;
  cout << "CLFFT Kernel execution time R2C Transform for size "<< N1 << "x" << N2 << " in <ms>:" << time_in_ms <<endl;

  /* Fetch results of calculations. */
  err = clEnqueueReadBuffer( queue, bufY, CL_TRUE, 0, complexSize * sizeof( *Y ), Y, 0, NULL, NULL );
  assert(err ==  CL_SUCCESS);

  /* Release OpenCL memory objects. */
  clReleaseMemObject( bufX );
  clReleaseMemObject( bufY );

  free(X);
  free(Y);

  /* Release the plan. */
  err = clfftDestroyPlan( &planHandle );
  assert(err ==  CL_SUCCESS);

  /* Release clFFT library. */
  clfftTeardown( );

  /* Release OpenCL working objects. */
  clReleaseCommandQueue( queue );
  clReleaseContext( ctx );

}

void hcfft_2d_c2r(size_t N1, size_t N2)
{
  FFTPlan plan;
  const hcfftDim dimension = HCFFT_2D;
  hcfftDirection dir = HCFFT_BACKWARD;
  hcfftResLocation location = HCFFT_OUTOFPLACE;
  hcfftResTransposed transposeType = HCFFT_NOTRANSPOSE;
  hcfftLibType libtype = HCFFT_C2RZ2D;
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t *ipStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t *opStrides = (size_t*)malloc(sizeof(size_t) * dimension);

  hcfftPlanHandle planhandle;
  hcfftPrecision precision = HCFFT_SINGLE;
  length[0] = N1;
  length[1] = N2;

  ipStrides[0] = 1;
  ipStrides[1] = 1 + length[0]/2;

  opStrides[0] = 1;
  opStrides[1] = length[0];

  size_t ipDistance = length[1] * (1 + length[0]/2);
  size_t opDistance = length[0] * length[1];

  int realsize, cmplexsize;
  realsize = length[0] * length[1];
  cmplexsize = length[1] * (1 + (length[0] / 2)) * 2;

  std::vector<accelerator> accs = accelerator::get_all();

  // Initialize host variables ----------------------------------------------
  float* ipHost = (float*)calloc(cmplexsize, sizeof(float));
  float* opHost = (float*)calloc(realsize, sizeof(float));

  for(int i = 0; i < cmplexsize / 2; i++) {
          ipHost[2 * i] = 2 * i + 1;
          ipHost[2 * i + 1] = 0;
  }

  float* ipDev = (float*)am_alloc(cmplexsize * sizeof(float), accs[1], 0);
  float* opDev = (float*)am_alloc(realsize * sizeof(float), accs[1], 0);

  // Copy input contents to device from host
  hc::am_copy(ipDev, ipHost, cmplexsize * sizeof(float));
  hc::am_copy(opDev, opHost, realsize * sizeof(float));

  hcfftStatus status = plan.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir, precision, libtype);
  if(status != HCFFT_SUCCEEDS) {
    cout << " Create plan error " << endl;
  }

  status = plan.hcfftSetAcclView(planhandle, accs[1].create_view());
  if(status != HCFFT_SUCCEEDS) {
    cout << " set accleration view error " << endl;
  }

  status = plan.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set plan error " << endl;
  }

  status = plan.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set result error " << endl;
  }

  status = plan.hcfftSetLayout(planhandle, HCFFT_HERMITIAN_INTERLEAVED, HCFFT_REAL);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set layout error " << endl;
  }

  status = plan.hcfftSetPlanInStride(planhandle, dimension, ipStrides );

  if(status != HCFFT_SUCCEEDS) {
    cout<<" hcfftSetPlanInStride error "<<endl;
  }

  status = plan.hcfftSetPlanOutStride(planhandle, dimension, opStrides );

  if(status != HCFFT_SUCCEEDS) {
    cout<<"hcfftSetPlanOutStride error "<<endl;
  }

  status = plan.hcfftSetPlanDistance(planhandle, ipDistance, opDistance );

  if(status != HCFFT_SUCCEEDS) {
    cout<<"hcfftSetPlanDistance error "<<endl;
  }

  status = plan.hcfftSetPlanScale(planhandle, dir, 1.0 );
  if( status != HCFFT_SUCCEEDS) {
    cout << " setplan scale error " << endl;;
  }

  status = plan.hcfftBakePlan(planhandle);
  if(status != HCFFT_SUCCEEDS) {
    cout << " bake plan error " << endl;
  }
  elapsed_pfe.clear();
  accs[1].get_default_view().wait();

  for(int i = 0 ; i < COUNT; i++)
  {
  starttimer = std::chrono::high_resolution_clock::now();

  status = plan.hcfftEnqueueTransform(planhandle, dir, ipDev, opDev, NULL);
  if(status != HCFFT_SUCCEEDS) {
    cout << " Transform error " << endl;
  }

  accs[1].get_default_view().wait();
  endtimer = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dur = endtimer - starttimer;
  if(i != 0)
    elapsed_pfe.push_back(dur);
  }

  double Avg_time = average(elapsed_pfe);
  double time_in_ms = Avg_time * 1e3;
  cout << "HCFFT Kernel execution time C2R Transform for size "<< N1 << "x" << N2 << " in <ms>:" << time_in_ms <<endl;

  // Copy Device output  contents back to host
  hc::am_copy(opHost, opDev, realsize * sizeof(float));

  status = plan.hcfftDestroyPlan(&planhandle);
  if(status != HCFFT_SUCCEEDS) {
    cout << " destroy plan error " << endl;
  }

  hc::am_free(ipDev);
  hc::am_free(opDev);

  free(ipHost);
  free(opHost);

  free(length);
  free(ipStrides);
  free(opStrides);
}

void clfft_2d_c2r(size_t N1, size_t N2)
{
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

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_2D;
  size_t clLengths[2] = { N1, N2};
  size_t ipStrides[2] = {1, 1 + N1/2};
  size_t ipDistance = N2 * (1+N1/2);
  size_t opStrides[2] = {1, N1};
  size_t  opDistance = N2*N1;
  cl_float scale = 1.0;

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  assert(err == CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );
  assert(err == CL_SUCCESS);

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  assert(err == CL_SUCCESS);

  queue = clCreateCommandQueue( ctx, device, 0, &err );
  assert(err ==  CL_SUCCESS);

  /* Setup clFFT. */
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  assert(err == CL_SUCCESS);

  err = clfftSetup(&fftSetup);
  assert(err == CL_SUCCESS);

  /* Allocate host & initialize data. */
  /* Only allocation shown for simplicity. */
  size_t realSize = N1 * N2;
  size_t complexSize = N2 * (1 + (N1 / 2)) * 2;

  X = (float *)calloc(realSize, sizeof(*X));
  Y = (float *)calloc(complexSize, sizeof(*Y));
  for(int i = 0; i < complexSize / 2; i++) {
          Y[2 * i] = 2 * i + 1;
          Y[2 * i + 1] = 0;
  }

  /* Prepare OpenCL memory objects and place data inside them. */
  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, realSize * sizeof(*X), NULL, &err );
  assert(err == CL_SUCCESS);

  bufY = clCreateBuffer( ctx, CL_MEM_READ_WRITE, complexSize * sizeof(*Y), NULL, &err );
  assert(err == CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
  realSize * sizeof( *X ), X, 0, NULL, NULL );
  assert(err == CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufY, CL_TRUE, 0,
  complexSize * sizeof( *Y ), Y, 0, NULL, NULL );
  assert(err == CL_SUCCESS);

  /*------------------------------------------------------C2R--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  assert(err == CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  assert(err == CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  assert(err == CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
  assert(err == CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  assert(err == CL_SUCCESS);

  err = clfftSetPlanInStride(planHandle, dim, ipStrides );
  assert(err == CL_SUCCESS);

  err = clfftSetPlanOutStride(planHandle, dim, opStrides );
  assert(err == CL_SUCCESS);

  err = clfftSetPlanDistance(planHandle, ipDistance, opDistance );
  assert(err == CL_SUCCESS);

  err = clfftSetPlanScale(planHandle, CLFFT_BACKWARD, scale );
  assert(err == CL_SUCCESS);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
  assert(err == CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  assert(err ==  CL_SUCCESS);

  elapsed_pfe.clear();

  for(int i = 0; i < COUNT; i++)
  {
  starttimer = std::chrono::high_resolution_clock::now();

  /* Execute the plan. */
  err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &bufY, &bufX, NULL);
  assert(err == CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  assert(err ==  CL_SUCCESS);

  endtimer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> dur = endtimer - starttimer;
  if(i != 0)
    elapsed_pfe.push_back(dur);
  }
  double Avg_time = average(elapsed_pfe);
  double time_in_ms = Avg_time * 1e3;
  cout << "CLFFT Kernel execution time C2R Transform for size "<< N1 << "x" << N2 << " in <ms>:" << time_in_ms <<endl;

  /* Fetch results of calculations. */
  err = clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, realSize * sizeof( *X ), X, 0, NULL, NULL );
  assert(err == CL_SUCCESS);

  /* Release OpenCL memory objects. */
  clReleaseMemObject( bufX );
  clReleaseMemObject( bufY );

  free(X);
  free(Y);

  /* Release the plan. */
  err = clfftDestroyPlan( &planHandle );
  assert(err == CL_SUCCESS);

  /* Release clFFT library. */
  clfftTeardown( );

  /* Release OpenCL working objects. */
  clReleaseCommandQueue( queue );
  clReleaseContext( ctx );
}

void hcfft_2d_c2c(size_t N1, size_t N2)
{
  FFTPlan plan;
  const hcfftDim dimension = HCFFT_2D;
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftResLocation location = HCFFT_OUTOFPLACE;
  hcfftResTransposed transposeType = HCFFT_NOTRANSPOSE;
  hcfftLibType libtype = HCFFT_C2CZ2Z;
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t *ipStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t *opStrides = (size_t*)malloc(sizeof(size_t) * dimension);

  hcfftPlanHandle planhandle;
  hcfftPrecision precision = HCFFT_SINGLE;
  length[0] = N1;
  length[1] = N2;

  ipStrides[0] = 1;
  ipStrides[1] = length[0];

  opStrides[0] = 1;
  opStrides[1] = length[0];

  size_t ipDistance = length[1] * length[0];
  size_t opDistance = length[1] * length[0];

  int size = length[0] * length[1] * 2;

  std::vector<accelerator> accs = accelerator::get_all();

  // Initialize host variables ----------------------------------------------
  float* ipHost = (float*)calloc(size, sizeof(float));
  float* opHost = (float*)calloc(size, sizeof(float));

  for(int  i = 0; i < N2 ; i++) {
    for(int  j = 0; j < N1 * 2; j++) {
      ipHost[i * N1 * 2 + j] = i * N1 * 2 + j + 1;
    }
  }

  float* ipDev = (float*)am_alloc(size * sizeof(float), accs[1], 0);
  float* opDev = (float*)am_alloc(size * sizeof(float), accs[1], 0);

  // Copy input contents to device from host
  hc::am_copy(ipDev, ipHost, size * sizeof(float));
  hc::am_copy(opDev, opHost, size * sizeof(float));

  hcfftStatus status = plan.hcfftCreateDefaultPlan (&planhandle, dimension, length, dir, precision, libtype);
  if(status != HCFFT_SUCCEEDS) {
    cout << " Create plan error " << endl;
  }

  status = plan.hcfftSetAcclView(planhandle, accs[1].create_view());
  if(status != HCFFT_SUCCEEDS) {
    cout << " set accleration view error " << endl;
  }

  status = plan.hcfftSetPlanPrecision(planhandle, precision);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set plan error " << endl;
  }

  status = plan.hcfftSetPlanTransposeResult(planhandle, transposeType);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set hcfftSetPlanTransposeResult error " << endl;
  }

  status = plan.hcfftSetResultLocation(planhandle, HCFFT_OUTOFPLACE);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set result error " << endl;
  }

  status = plan.hcfftSetPlanInStride(planhandle, dimension, ipStrides );

  if(status != HCFFT_SUCCEEDS) {
    cout<<" hcfftSetPlanInStride error "<<endl;
  }

  status = plan.hcfftSetPlanOutStride(planhandle, dimension, opStrides );

  if(status != HCFFT_SUCCEEDS) {
    cout<<"hcfftSetPlanOutStride error "<<endl;
  }

  status = plan.hcfftSetPlanDistance(planhandle, ipDistance, opDistance );

  if(status != HCFFT_SUCCEEDS) {
    cout<<"hcfftSetPlanDistance error "<<endl;
  }

  /*---------------------R2C--------------------------------------*/
  status = plan.hcfftSetLayout(planhandle, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    cout << " set layout error " << endl;
  }

  status = plan.hcfftBakePlan(planhandle);

  if(status != HCFFT_SUCCEEDS) {
    cout << " bake plan error " << endl;
  }

  elapsed_pfe.clear();
  accs[1].get_default_view().wait();
  for(int i = 0; i < COUNT; i++)
  {
  starttimer = std::chrono::high_resolution_clock::now();

  status = plan.hcfftEnqueueTransform(planhandle, dir, ipDev, opDev, NULL);
  if(status != HCFFT_SUCCEEDS) {
    cout << " Transform error " << endl;
  }
  accs[1].get_default_view().wait();

  endtimer = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dur = endtimer - starttimer;
  if(i != 0)
    elapsed_pfe.push_back(dur);
  }

  double Avg_time = average(elapsed_pfe);
  double time_in_ms = Avg_time * 1e3;
  cout << "HCFFT Kernel execution time C2C Transform for size "<< N1 << "x" << N2 << " in <ms>:" << time_in_ms <<endl;

  // Copy Device output  contents back to host
  hc::am_copy(opHost, opDev, size * sizeof(float));

  status = plan.hcfftDestroyPlan(&planhandle);
  if(status != HCFFT_SUCCEEDS) {
    cout << " destroy plan error " << endl;
  }

  hc::am_free(ipDev);
  hc::am_free(opDev);

  free(ipHost);
  free(opHost);

  free(length);
  free(ipStrides);
  free(opStrides);
}

void clfft_2d_c2c(size_t N1, size_t N2)
{
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

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_2D;
  size_t clLengths[2] = { N1, N2};
  size_t ipStrides[2] = {1, N1};
  size_t ipDistance = N2 * N1;
  size_t opStrides[2] = {1, N1};
  size_t  opDistance = N2 * N1;

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  assert(err == CL_SUCCESS);

  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );
  assert(err == CL_SUCCESS);

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  assert(err == CL_SUCCESS);

  queue = clCreateCommandQueue( ctx, device, 0, &err );
  assert(err == CL_SUCCESS);

  /* Setup clFFT. */
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  assert(err == CL_SUCCESS);

  err = clfftSetup(&fftSetup);
  assert(err == CL_SUCCESS);

  /* Allocate host & initialize data. */
  /* Only allocation shown for simplicity. */
  size_t size = N1 * N2 * 2;

  X = (float *)calloc(size, sizeof(*X));
  Y = (float *)calloc(size, sizeof(*Y));
  for(int i = 0; i < size / 2; i++) {
          X[2 * i] = 2 * i + 1;
          X[2 * i + 1] = 0;
  }

  /* Prepare OpenCL memory objects and place data inside them. */
  bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, size * sizeof(*X), NULL, &err );
  assert(err == CL_SUCCESS);

  bufY = clCreateBuffer( ctx, CL_MEM_READ_WRITE, size * sizeof(*Y), NULL, &err );
  assert(err == CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,
  size * sizeof( *X ), X, 0, NULL, NULL );
  assert(err == CL_SUCCESS);

  err = clEnqueueWriteBuffer( queue, bufY, CL_TRUE, 0,
  size * sizeof( *Y ), Y, 0, NULL, NULL );
  assert(err == CL_SUCCESS);

  /*------------------------------------------------------C2C--------------------------------------------------------------------*/
  /* Create a default plan for a complex FFT. */

  err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
  assert(err == CL_SUCCESS);

  /* Set plan parameters. */
  err = clfftSetPlanTransposeResult(planHandle, CLFFT_NOTRANSPOSE);
  assert(err == CL_SUCCESS);

  err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
  assert(err == CL_SUCCESS);

  err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
  assert(err == CL_SUCCESS);

  err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
  assert(err == CL_SUCCESS);

  err = clfftSetPlanInStride(planHandle, dim, ipStrides );
  assert(err == CL_SUCCESS);

  err = clfftSetPlanOutStride(planHandle, dim, opStrides );
  assert(err == CL_SUCCESS);

  err = clfftSetPlanDistance(planHandle, ipDistance, opDistance );
  assert(err == CL_SUCCESS);

  /* Bake the plan. */
  err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
  assert(err == CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  assert(err ==  CL_SUCCESS);

  elapsed_pfe.clear();

  for(int i = 0; i < COUNT; i++)
  {
  starttimer = std::chrono::high_resolution_clock::now();

  /* Execute the plan. */
  err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, &bufY, NULL);
  assert(err == CL_SUCCESS);

  /* Wait for calculations to be finished. */
  err = clFinish(queue);
  assert(err ==  CL_SUCCESS);

  endtimer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> dur = endtimer - starttimer;
  if(i != 0)
    elapsed_pfe.push_back(dur);
  }

  double Avg_time = average(elapsed_pfe);
  double time_in_ms = Avg_time * 1e3;
  cout << "CLFFT Kernel execution time C2C Transform for size "<< N1 << "x" << N2 << " in <ms>:" << time_in_ms <<endl;

  /* Fetch results of calculations. */
  err = clEnqueueReadBuffer( queue, bufY, CL_TRUE, 0, size * sizeof( *Y ), Y, 0, NULL, NULL );
  assert(err == CL_SUCCESS);

  /* Release OpenCL memory objects. */
  clReleaseMemObject( bufX );
  clReleaseMemObject( bufY );

  free(X);
  free(Y);

  /* Release the plan. */
  err = clfftDestroyPlan( &planHandle );
  assert(err == CL_SUCCESS);

  /* Release clFFT library. */
  clfftTeardown( );

  /* Release OpenCL working objects. */
  clReleaseCommandQueue( queue );
  clReleaseContext( ctx ); 
}

int main(int argc, char* argv[])
{
  size_t N1, N2;
  N1 = atoi(argv[1]);
  N2 = atoi(argv[2]);

  hcfft_2d_r2c(N1, N2);
  clfft_2d_r2c(N1, N2); 

  hcfft_2d_c2r(N1, N2);
  clfft_2d_c2r(N1, N2); 

  hcfft_2d_c2c(N1, N2);
  clfft_2d_c2c(N1, N2);

}
