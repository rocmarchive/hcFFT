#include "hipfft.h"

#ifdef __cplusplus
extern "C" {
#endif

hipfftResult hipCUFFTResultToHIPFFTResult(cufftResult cuResult) 
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

cufftType hipHIPFFTTypeToCUFFTType(hipfftType hipType) 
{
   switch(hipType) 
   {
    case HIPFFT_R2C:
        return CUFFT_R2C;
    case HIPFFT_C2R:
        return CUFFT_C2R;
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

int hipHIPFFTDirectionToCUFFTDirection(hipfftDirection hipDirection)
{
    switch(hipDirection)
    {
        case HIPFFT_FORWARD:
          return CUFFT_FORWARD;
        case HIPFFT_INVERSE:
          return CUFFT_INVERSE;
        default:
          throw "Unimplemented direction";
    }
}

hipfftResult hipfftCreate(hipfftHandle *plan){
    return hipCUFFTResultToHIPFFTResult(cufftCreate(plan));
}


hipfftResult hipfftDestroy(hipfftHandle plan){
    return hipCUFFTResultToHIPFFTResult(cufftDestroy(plan));
}


hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream){
    return hipCUFFTResultToHIPFFTResult(cufftSetStream(plan, stream));
}

/*hipFFT Basic Plans*/

hipfftResult hipfftPlan1d(hipfftHandle *plan, int nx, hipfftType type, int batch){
    return hipCUFFTResultToHIPFFTResult(cufftPlan1d(plan, nx, hipHIPFFTTypeToCUFFTType(type), batch));
}

hipfftResult hipfftPlan2d(hipfftHandle *plan, int nx, int ny, hipfftType type){
    return hipCUFFTResultToHIPFFTResult(cufftPlan2d(plan, nx, ny, hipHIPFFTTypeToCUFFTType(type)));
}


hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type){
    return hipCUFFTResultToHIPFFTResult(cufftPlan3d(plan, nx, ny, nz, hipHIPFFTTypeToCUFFTType(type)));
}

hipfftResult hipfftPlanMany(hipfftHandle *plan, int rank, int *n, int *inembed,int istride, 
                                          int idist, int *onembed, int ostride,
                                          int odist, hipfftType type, int batch){

    return hipCUFFTResultToHIPFFTResult(cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, 
                                                       ostride, odist, hipHIPFFTTypeToCUFFTType(type), batch));
}

/*hipFFT Extensible Plans*/

hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize){
    return hipCUFFTResultToHIPFFTResult(cufftMakePlan1d(plan, nx, hipHIPFFTTypeToCUFFTType(type), batch, workSize));
}


hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize){
    return hipCUFFTResultToHIPFFTResult(cufftMakePlan2d(plan, nx, ny, hipHIPFFTTypeToCUFFTType(type), workSize));
}


hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize){
    return hipCUFFTResultToHIPFFTResult(cufftMakePlan3d(plan, nx, ny, nz, hipHIPFFTTypeToCUFFTType(type), workSize));
}



hipfftResult hipfftMakePlanMany(hipfftHandle plan, int rank, int *n, int *inembed, int istride, 
                                              int idist, int *onembed, int ostride, int odist, hipfftType type, 
                                              int batch, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, 
                                                         odist, hipHIPFFTTypeToCUFFTType(type), batch, workSize));
}



hipfftResult hipfftMakePlanMany64(hipfftHandle plan, int rank, long long int *n, 
                                                long long int *inembed, long long int istride, long long int idist, 
                                                long long int *onembed, long long int ostride, long long int odist, 
                                                hipfftType type, long long int batch, size_t *workSize){

  return hipCUFFTResultToHIPFFTResult(cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, 
                                                          odist, hipHIPFFTTypeToCUFFTType(type), batch, workSize));

}

/*hipFFT Estimated Size of Work Area*/

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftEstimate1d(nx, hipHIPFFTTypeToCUFFTType(type), batch, workSize));
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftEstimate2d(nx, ny, hipHIPFFTTypeToCUFFTType(type), workSize));
}


hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftEstimate3d(nx, ny, nz, hipHIPFFTTypeToCUFFTType(type), workSize));
}


hipfftResult hipfftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, 
                                              int ostride, int odist, hipfftType type, int batch, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, 
                                                         odist, hipHIPFFTTypeToCUFFTType(type), batch, workSize));
}

/*hipFFT Refined Estimated Size of Work Area*/

hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftGetSize1d(plan, nx, hipHIPFFTTypeToCUFFTType(type), batch, workSize));
}


hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftGetSize2d(plan, nx, ny, hipHIPFFTTypeToCUFFTType(type), workSize));
}


hipfftResult hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, 
                                           size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftGetSize3d(plan, nx, ny, nz, hipHIPFFTTypeToCUFFTType(type), workSize));
}

hipfftResult hipfftGetSizeMany(hipfftHandle plan, int rank, int *n, int *inembed,
                                             int istride, int idist, int *onembed, int ostride,
                                             int odist, hipfftType type, int batch, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftGetSizeMany(plan, rank, n, inembed, istride, idist, onembed, 
                                                     ostride, odist, hipHIPFFTTypeToCUFFTType(type), batch, workSize));
}

hipfftResult hipfftGetSizeMany64(hipfftHandle plan, int rank, long long int *n, 
                                              long long int *inembed, long long int istride, long long int idist, 
                                              long long int *onembed, long long int ostride, long long int odist, 
                                              hipfftType type, long long int batch, size_t *workSize){

  return hipCUFFTResultToHIPFFTResult(cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, 
                                                         onembed, ostride, odist, hipHIPFFTTypeToCUFFTType(type), 
                                                         batch, workSize));
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t *workSize){
  return hipCUFFTResultToHIPFFTResult(cufftGetSize(plan, workSize));
}

/*hipFFT Caller Allocated Work Area Support*/

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate){
  return hipCUFFTResultToHIPFFTResult(cufftSetAutoAllocation(plan, autoAllocate));
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void *workArea){
  return hipCUFFTResultToHIPFFTResult(cufftSetWorkArea(plan, workArea));
}

/*hipFFT Execution*/

hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftComplex *odata, int direction){
    return hipCUFFTResultToHIPFFTResult(cufftExecC2C(plan, idata, odata, direction));
}

hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, 
                                         hipfftDoubleComplex *odata, int direction){
    return hipCUFFTResultToHIPFFTResult(cufftExecZ2Z(plan, idata, odata, direction));
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata, 
                                         hipfftComplex *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecR2C(plan, idata, odata));
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata, 
                                         hipfftDoubleComplex *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecD2Z(plan, idata, odata));
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata, 
                                         hipfftReal *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecC2R(plan, idata, odata));
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex *idata, 
                                         hipfftDoubleReal *odata){
    return hipCUFFTResultToHIPFFTResult(cufftExecZ2D(plan, idata, odata));
}

#ifdef __cplusplus
}
#endif
