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

#include <cuda_runtime_api.h>
#include <cusparse.h>

#ifdef cplusplus
extern "C" {
#endif

typedef cufftHandle hipfftHandle;
typedef cudaStream_t hipStream_t;
typedef cufftComplex hipfftComplex;
typedef cufftDoubleComplex hipfftDoubleComplex;
typedef cufftReal  hipfftReal;
typedef cufftDoubleReal hipfftDoubleReal;

inline static hipfftResult hipCUFFTResultToHIPFFTResult(cufftResult cuResult) 
{
   switch(cuResult) 
   {
    case CUFFT_SUCCESS:
        return HIPFFT_SUCCESS;
    case CUFFT_INVALID_PLAN:
        return HIPFFT_INVALID_PLAN;
    case CUFFT_ALLOC_FAILED:
        return HIPFFT_ALLOC_FAILED;
    case CUFFT_INVALID_TYPE:
        return HIPFFT_INVALID_TYPE;
    case CUFFT_INVALID_VALUE:
        return HIPFFT_INVALID_VALUE;
    case CUFFT_INTERNAL_ERROR:
        return HIPFFT_INTERNAL_ERROR;
    case CUFFT_EXEC_FAILED:
        return HIPFFT_EXEC_FAILED;
    case CUFFT_SETUP_FAILED:
        return HIPFFT_SETUP_FAILED;
    case CUFFT_INVALID_SIZE:
        return HIPFFT_INVALID_SIZE;
    case CUFFT_UNALIGNED_DATA:
        return HIPFFT_UNALIGNED_DATA;
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return HIPFFT_INCOMPLETE_PARAMETER_LIST;
    case CUFFT_INVALID_DEVICE:
        return HIPFFT_INVALID_DEVICE;
    case CUFFT_PARSE_ERROR:
        return HIPFFT_PARSE_ERROR;
    case CUFFT_NO_WORKSPACE:
        return HIPFFT_NO_WORKSPACE;
    default:
         throw "Unimplemented status";
   }
}

inline static hipfftType hipHIPFFTTypeToCUFFTType(hipfftType hipType) 
{
   switch(hipType) 
   {
    case HIPFFT_R2C:
        return CUFFT_R2C;
    case HIPFFT_C2R:
        return CUFFT_C2R
    case HIPFFT_C2C:
        return CUFFT_C2C;
    case HIPFFT_D2Z:
        return CUFFT_D2Z;
    case HIPFFT_Z2D:
        return CUFFT_Z2D;
    case HIPFFT_Z2Z:
        return CUFFT_Z2Z;
    default:
        throw "Unimplemented Type";
  }
}


inline static hipfftResult hipfftCreate(hipfftHandle *plan){
    return hipCUFFTResultToHIPFFTResult(cufftCreate(&plan));
}

inline static hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream){
    return hipCUFFTResultToHIPFFTResult(cufftSetStream(plan, stream));
}

/*hipFFT Basic Plans*/

inline static hipfftResult hipfftPlan1d(hipfftHandle *plan, int nx, hipfftType type, int batch){
    return hipCUFFTResultToHIPFFTResult(cufftPlan1d(&plan, nx, hipHIPFFTTypeToCUFFTType(type), batch));
}

inline static hipfftResult hipfftPlan2d(hipfftHandle *plan, int nx, int ny, hipfftType type){
    return hipCUFFTResultToHIPFFTResult(cufftPlan2d(&plan, nx, ny, hipHIPFFTTypeToCUFFTType(type)));
}

inline static hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type){
    return hipCUFFTResultToHIPFFTResult(cufftPlan3d(&plan, nx, ny, nz, hipHIPFFTTypeToCUFFTType(type)));
}

inline static hipfftResult hipfftDestroy(hipfftHandle plan){
    return hipCUFFTResultToHIPFFTResult(cufftDestroy(plan));
}

/*hipFFT Execution*/

inline static hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftComplex *odata, int direction){
    return hipCUFFTResultToHIPFFTResult(cufftExecC2C(plan, idata, odata, direction));
}

inline static hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, 
                                         hipfftDoubleComplex *odata, int direction){
    return hipCUFFTResultToHIPFFTResult(cufftExecZ2Z(plan, idata, odata, direction));
}

inline static hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata, 
                                         hipfftComplex *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecR2C(plan, idata, odata));
}

inline static hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata, 
                                         hipfftDoubleComplex *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecD2Z(plan, idata, odata));
}

inline static hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftReal *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecC2R(plan, idata, odata));
}

inline static hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftReal *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecZ2D(plan, idata, odata));
}

#ifdef __cplusplus
}
#endif


