/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <hip/hip_runtime_api.h>
#include <hcfft.h>
#include <hip/hip_complex.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef hcfftHandle hipfftHandle;
typedef hipComplex hipfftComplex;
typedef hipDoubleComplex hipfftDoubleComplex;
typedef hcfftReal  hipfftReal;
typedef hcfftDoubleReal hipfftDoubleReal;

 hipfftResult hipHCFFTResultToHIPFFTResult(hcfftResult hcResult);

 hcfftType hipHIPFFTTypeToHCFFTType(hipfftType hipType);

 int hipHIPFFTDirectionToHCFFTDirection(int hipDirection);

 hipfftResult hipfftCreate(hipfftHandle *plan);

 hipfftResult hipfftDestroy(hipfftHandle plan);

 hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream);

/*hipFFT Basic Plans*/

 hipfftResult hipfftPlan1d(hipfftHandle *plan, int nx, hipfftType type, int batch);

 hipfftResult hipfftPlan2d(hipfftHandle *plan, int nx, int ny, hipfftType type);

 hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type);

 hipfftResult hipfftPlanMany(hipfftHandle *plan, int rank, int *n, int *inembed,int istride, 
                                          int idist, int *onembed, int ostride,
                                          int odist, hipfftType type, int batch);

/*hipFFT Extensible Plans*/

 hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize);

 hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize);

 hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize);


 hipfftResult hipfftMakePlanMany(hipfftHandle plan, int rank, int *n, int *inembed, int istride, 
                                              int idist, int *onembed, int ostride, int odist, hipfftType type, 
                                              int batch, size_t *workSize);

 hipfftResult hipfftMakePlanMany64(hipfftHandle plan, int rank, long long int *n, 
                                                long long int *inembed, long long int istride, long long int idist, 
                                                long long int *onembed, long long int ostride, long long int odist, 
                                                hipfftType type, long long int batch, size_t *workSize);

/*hipFFT Estimated Size of Work Area*/

 hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t *workSize);

 hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t *workSize);

 hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t *workSize);

 hipfftResult hipfftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, 
                                              int ostride, int odist, hipfftType type, int batch, size_t *workSize);

/*hipFFT Refined Estimated Size of Work Area*/

 hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize);

 hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize);

 hipfftResult hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, 
                                           size_t *workSize);

 hipfftResult hipfftGetSizeMany(hipfftHandle plan, int rank, int *n, int *inembed,
                                             int istride, int idist, int *onembed, int ostride,
                                             int odist, hipfftType type, int batch, size_t *workSize);

 hipfftResult hipfftGetSizeMany64(hipfftHandle plan, int rank, long long int *n, 
                                              long long int *inembed, long long int istride, long long int idist, 
                                              long long int *onembed, long long int ostride, long long int odist, 
                                              hipfftType type, long long int batch, size_t *workSize);

 hipfftResult hipfftGetSize(hipfftHandle plan, size_t *workSize);

/*hipFFT Caller Allocated Work Area Support*/

 hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate);

 hipfftResult hipfftSetWorkArea(hipfftHandle plan, void *workArea);


/*hipFFT Execution*/

 hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftComplex *odata, int direction);

 hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, 
                                         hipfftDoubleComplex *odata, int direction);

 hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata, 
                                         hipfftComplex *odata);

 hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata, 
                                         hipfftDoubleComplex *odata);

 hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftReal *odata);

 hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex *idata, 
                                         hipfftDoubleReal *odata);

#ifdef __cplusplus
}
#endif
