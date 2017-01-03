#include <iostream>
#include "hipfft.h"

#ifdef __cplusplus
extern "C" {
#endif

    hipfftResult hipfftCreate(hipfftHandle *plan){
    return hipHCFFTResultToHIPFFTResult(hcfftCreate(plan));
}

    hipfftResult hipfftDestroy(hipfftHandle plan){
    return hipHCFFTResultToHIPFFTResult(hcfftDestroy(plan));
}

    hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream){
    return HIPFFT_RESULT_NOT_SUPPORTED;
}

/*hipFFT Basic Plans*/

    hipfftResult hipfftPlan1d(hipfftHandle *plan, int nx, hipfftType type, int batch){
    return hipHCFFTResultToHIPFFTResult(hcfftPlan1d(plan, nx, hipHIPFFTTypeToHCFFTType(type))); 
}

    hipfftResult hipfftPlan2d(hipfftHandle *plan, int nx, int ny, hipfftType type){
    return hipHCFFTResultToHIPFFTResult(hcfftPlan2d(plan, nx, ny, hipHIPFFTTypeToHCFFTType(type)));
}

    hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type){
    return hipHCFFTResultToHIPFFTResult(hcfftPlan3d(plan, nx, ny, nz, hipHIPFFTTypeToHCFFTType(type)));
}

    hipfftResult hipfftPlanMany(hipfftHandle *plan, int rank, int *n, int *inembed,int istride, 
                                          int idist, int *onembed, int ostride,
                                          int odist, hipfftType type, int batch){
    return HIPFFT_RESULT_NOT_SUPPORTED;
}

/*hipFFT Extensible Plans*/

    hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize){
    return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize){
    return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize){
    return HIPFFT_RESULT_NOT_SUPPORTED;
}


    hipfftResult hipfftMakePlanMany(hipfftHandle plan, int rank, int *n, int *inembed, int istride, 
                                              int idist, int *onembed, int ostride, int odist, hipfftType type, 
                                              int batch, size_t *workSize){
  	return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftMakePlanMany64(hipfftHandle plan, int rank, long long int *n, 
                                                long long int *inembed, long long int istride, long long int idist, 
                                                long long int *onembed, long long int ostride, long long int odist, 
                                                hipfftType type, long long int batch, size_t *workSize){
  	return HIPFFT_RESULT_NOT_SUPPORTED; 
}

/*hipFFT Estimated Size of Work Area*/

    hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t *workSize){
  	return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t *workSize){
  	return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t *workSize){
  	return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, 
                                              int ostride, int odist, hipfftType type, int batch, size_t *workSize){
  	return HIPFFT_RESULT_NOT_SUPPORTED;
}

/*hipFFT Refined Estimated Size of Work Area*/

    hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, 
                                            size_t *workSize){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftGetSizeMany(hipfftHandle plan, int rank, int *n, int *inembed,
                                             int istride, int idist, int *onembed, int ostride,
                                             int odist, hipfftType type, int batch, size_t *workSize){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftGetSizeMany64(hipfftHandle plan, int rank, long long int *n, 
                                              long long int *inembed, long long int istride, long long int idist, 
                                              long long int *onembed, long long int ostride, long long int odist, 
                                              hipfftType type, long long int batch, size_t *workSize){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftGetSize(hipfftHandle plan, size_t *workSize){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}

/*hipFFT Caller Allocated Work Area Support*/

    hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}

    hipfftResult hipfftSetWorkArea(hipfftHandle plan, void *workArea){
  return HIPFFT_RESULT_NOT_SUPPORTED;
}


/*hipFFT Execution*/

    hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftComplex *odata, int direction){
    return hipHCFFTResultToHIPFFTResult(hcfftExecC2C(plan, (hcfftComplex *)idata, (hcfftComplex *)odata, hipHIPFFTDirectionToHCFFTDirection(direction)));
}

    hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, 
                                         hipfftDoubleComplex *odata, int direction){
    return hipHCFFTResultToHIPFFTResult(hcfftExecZ2Z(plan, (hcfftDoubleComplex *)idata, (hcfftDoubleComplex *)odata, hipHIPFFTDirectionToHCFFTDirection(direction)));
}

    hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata, 
                                         hipfftComplex *odata){
    return hipHCFFTResultToHIPFFTResult(hcfftExecR2C(plan, idata, (hcfftComplex *)odata));
}

    hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata, 
                                         hipfftDoubleComplex *odata){
    return hipHCFFTResultToHIPFFTResult(hcfftExecD2Z(plan, idata, (hcfftDoubleComplex *)odata));
}

    hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftReal *odata){
    return hipHCFFTResultToHIPFFTResult(hcfftExecC2R(plan, (hcfftComplex *)idata, odata));
}

    hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex *idata, 
                                         hipfftDoubleReal *odata){
    return hipHCFFTResultToHIPFFTResult(hcfftExecZ2D(plan, (hcfftDoubleComplex *)idata, odata));
}


#ifdef __cplusplus
}
#endif
