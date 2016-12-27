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

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified 
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//!  This is the master include file for hipfft, wrapping around hcfft and cufft
//

#pragma once

typedef enum hipfftType_t {
    HIPFFT_R2C = 0x2a,  // Real to complex (interleaved) 
    HIPFFT_C2R = 0x2c,  // Complex (interleaved) to real 
    HIPFFT_C2C = 0x29,  // Complex to complex (interleaved) 
    HIPFFT_D2Z = 0x6a,  // Double to double-complex (interleaved) 
    HIPFFT_Z2D = 0x6c,  // Double-complex (interleaved) to double 
    HIPFFT_Z2Z = 0x69   // Double-complex to double-complex (interleaved)
}hipfftType;

typedef enum hipfftResult_t {
  HIPFFT_SUCCESS        = 0,  //  The hipFFT operation was successful
  HIPFFT_INVALID_PLAN   = 1,  //  hipFFT was passed an invalid plan handle
  HIPFFT_ALLOC_FAILED   = 2,  //  hipFFT failed to allocate GPU or CPU memory
  HIPFFT_INVALID_TYPE   = 3,  //  No longer used
  HIPFFT_INVALID_VALUE  = 4,  //  User specified an invalid pointer or parameter
  HIPFFT_INTERNAL_ERROR = 5,  //  Driver or internal hipFFT library error
  HIPFFT_EXEC_FAILED    = 6,  //  Failed to execute an FFT on the GPU
  HIPFFT_SETUP_FAILED   = 7,  //  The hipFFT library failed to initialize
  HIPFFT_INVALID_SIZE   = 8,  //  User specified an invalid transform size
  HIPFFT_UNALIGNED_DATA = 9,  //  No longer used
  HIPFFT_INCOMPLETE_PARAMETER_LIST = 10, //  Missing parameters in call
  HIPFFT_INVALID_DEVICE = 11, //  Execution of a plan was on different GPU than plan creation
  HIPFFT_PARSE_ERROR    = 12, //  Internal plan database error
  HIPFFT_NO_WORKSPACE   = 13,  //  No workspace has been provided prior to plan execution
  HIPFFT_RESULT_NOT_SUPPORTED = 14
}hipfftResult;

typedef enum hipfftDirection_ {
  HIPFFT_FORWARD = -1,
  HIPFFT_INVERSE = 1,
} hipfftDirection;

// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:

#if defined(__HIP_PLATFORM_HCC__) and not defined (__HIP_PLATFORM_NVCC__) 
#include <hcc_detail/hip_fft.h>
#elif defined(__HIP_PLATFORM_NVCC__) and not defined (__HIP_PLATFORM_HCC__)
#include <nvcc_detail/hip_fft.h>
#else 
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif 


