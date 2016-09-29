#include "hcfft.h"

// Global Static plan object
thread_local FFTPlan planObject;

/* Function hcfftXtSetGPUs()
Returns GPUs are to be used with the plan
*/
hcfftResult hcfftXtSetGPUs(hc::accelerator &acc) {
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();

  if(accs.size() == 0) {
    std::wcout << "There is no acclerator!\n";
    // Since this case is to test on GPU device, skip if there is CPU only
    return HCFFT_SETUP_FAILED;
  }

  assert(accs.size() && "Number of Accelerators == 0!");
  acc = accs[1];
  return HCFFT_SUCCESS;
}

/* Function hcfftSetStream()
Associate FFT Plan with an accelerator_view
*/
hcfftResult hcfftSetStream(hcfftHandle*&plan, hc::accelerator_view &acc_view) {
  hcfftStatus status = planObject.hcfftSetAcclView(*plan, acc_view);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  return HCFFT_SUCCESS;
}

/* Function hcfftCreate()
Creates only an opaque handle, and allocates small data structures on the host.
*/
hcfftResult hcfftCreate(hcfftHandle*&plan) {
  if(plan == NULL) {
    // create new plan
    plan = new hcfftHandle;
    return HCFFT_SUCCESS;
  } else {
    plan = NULL;
    plan = new hcfftHandle;
    return HCFFT_SUCCESS;
  }
}

/******************************************************************************************************************
 * <i>  Function hcfftPlan1d()
   Description:
       Creates a 1D FFT plan configuration for a specified signal size and data type. The batch input parameter tells
   hcFFT how many 1D transforms to configure.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan  Pointer to a hcfftHandle object
   #2 nx  The transform size (e.g. 256 for a 256-point FFT)
   #3 type  The transform data type (e.g., HCFFT_C2C for single precision complex to complex)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan  Contains a hcFFT 1D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED   The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE  One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR An internal driver error was detected.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE   The nx parameter is not a supported size.
 ***********************************************************************************************************************
 */

hcfftResult hcfftPlan1d(hcfftHandle* &plan, int nx, hcfftType type) {
  // Set dimension as 1D
  hcfftDim dimension = HCFFT_1D;
  // Check the input type and set appropriate direction and precision
  hcfftDirection direction;
  hcfftPrecision precision;

  switch (type) {
    case HCFFT_R2C:
      precision = HCFFT_SINGLE;
      direction = HCFFT_FORWARD;
      break;

    case HCFFT_C2R:
      precision = HCFFT_SINGLE;
      direction = HCFFT_BACKWARD;
      break;

    case HCFFT_C2C:
      precision = HCFFT_SINGLE;
      direction = HCFFT_BOTH;
      break;

    case HCFFT_D2Z:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_FORWARD;
      break;

    case HCFFT_Z2D:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_BACKWARD;
      break;

    case HCFFT_Z2Z:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_BOTH;
      break;

    default:
      // Invalid type
      return HCFFT_INVALID_VALUE;
  }

  // length array to bookkeep dimension info
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* ipStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* opStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t ipDistance, opDistance;
  float scale;

  if ( nx < 0 ) {
    // invalid size
    return HCFFT_INVALID_SIZE;
  } else {
    length[0] = nx;
  }

  // Allocate Rawplan
  hcfftResult res = hcfftCreate(plan);

  if(res != HCFFT_SUCCESS) {
    return HCFFT_ALLOC_FAILED;
  }

  hc::accelerator acc;
  res = hcfftXtSetGPUs(acc);

  if(res != HCFFT_SUCCESS) {
    return HCFFT_SETUP_FAILED;
  }

  hcfftLibType libType = ((type == HCFFT_R2C || type == HCFFT_D2Z) ? HCFFT_R2CD2Z : (type == HCFFT_C2R || type == HCFFT_Z2D) ? HCFFT_C2RZ2D : (type == HCFFT_C2C || type == HCFFT_Z2Z ) ? HCFFT_C2CZ2Z : (hcfftLibType)0);

  switch (libType) {
    case HCFFT_R2CD2Z:
      ipStrides[0] = 1;
      opStrides[0] = 1;
      ipDistance = nx;
      opDistance = 1 + nx / 2;
      break;

    case HCFFT_C2RZ2D:
      ipStrides[0] = 1;
      opStrides[0] = 1;
      ipDistance = 1 + nx / 2;
      opDistance = nx;
      scale = 1.0;
      break;

    case HCFFT_C2CZ2Z:
      ipStrides[0] = 1;
      opStrides[0] = 1;
      ipDistance = nx;
      opDistance = nx;
      break;

    default:
      // Invalid type
      return HCFFT_INVALID_VALUE;
  }

  hcfftStatus status = planObject.hcfftCreateDefaultPlan (plan, dimension, length, direction, precision, libType);

  if ( status == HCFFT_ERROR || status == HCFFT_INVALID ) {
    return HCFFT_INVALID_VALUE;
  }

  // Default options
  // set certain properties of plan with default values
  // Set Precision
  status = planObject.hcfftSetPlanPrecision(*plan, precision);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  // Set Transpose type
  status = planObject.hcfftSetPlanTransposeResult(*plan, HCFFT_NOTRANSPOSE);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  // Set Result location data layout
  status = planObject.hcfftSetResultLocation(*plan, HCFFT_OUTOFPLACE);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanInStride(*plan, dimension, ipStrides);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanOutStride(*plan, dimension, opStrides);

  if( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanDistance(*plan, ipDistance, opDistance);

  if( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  if ( libType == HCFFT_C2RZ2D) {
    status = planObject.hcfftSetPlanScale(*plan, direction, scale );

    if( status != HCFFT_SUCCEEDS) {
      return HCFFT_SETUP_FAILED;
    }
  }

  return HCFFT_SUCCESS;
}


/*
 * <ii> Function hcfftPlan2d()
   Description:
      Creates a 2D FFT plan configuration according to specified signal sizes and data type.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan  Pointer to a hcfftHandle object
   #2 nx  The transform size in the x dimension (number of rows)
   #3 ny  The transform size in the y dimension (number of columns)
   #4 type  The transform data type (e.g., HCFFT_C2R for single precision complex to real)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan  Contains a hcFFT 2D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED   The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE  One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR An internal driver error was detected.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE   Either or both of the nx or ny parameters is not a supported sizek.
*/

hcfftResult hcfftPlan2d(hcfftHandle*&plan, int nx, int ny, hcfftType type) {
  // Set dimension as 2D
  hcfftDim dimension = HCFFT_2D;
  // Check the input type and set appropriate direction and precision
  hcfftDirection direction;
  hcfftPrecision precision;

  switch (type) {
    case HCFFT_R2C:
      precision = HCFFT_SINGLE;
      direction = HCFFT_FORWARD;
      break;

    case HCFFT_C2R:
      precision = HCFFT_SINGLE;
      direction = HCFFT_BACKWARD;
      break;

    case HCFFT_C2C:
      precision = HCFFT_SINGLE;
      direction = HCFFT_BOTH;
      break;

    case HCFFT_D2Z:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_FORWARD;
      break;

    case HCFFT_Z2D:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_BACKWARD;
      break;

    case HCFFT_Z2Z:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_BOTH;
      break;

    default:
      // Invalid type
      return HCFFT_INVALID_VALUE;
  }

  // length array to bookkeep dimension info
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* ipStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* opStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t ipDistance, opDistance;
  float scale;

  if (nx < 0 || ny < 0) {
    // invalid size
    return HCFFT_INVALID_SIZE;
  } else {
    length[0] = nx;
    length[1] = ny;
  }

  // Allocate Rawplan
  hcfftResult res = hcfftCreate(plan);

  if(res != HCFFT_SUCCESS) {
    return HCFFT_ALLOC_FAILED;
  }

  hc::accelerator acc;
  res = hcfftXtSetGPUs(acc);

  if(res != HCFFT_SUCCESS) {
    return HCFFT_SETUP_FAILED;
  }

  hcfftLibType libType = ((type == HCFFT_R2C || type == HCFFT_D2Z) ? HCFFT_R2CD2Z : (type == HCFFT_C2R || type == HCFFT_Z2D) ? HCFFT_C2RZ2D : (type == HCFFT_C2C || type == HCFFT_Z2Z ) ? HCFFT_C2CZ2Z : (hcfftLibType)0);

  switch (libType) {
    case HCFFT_R2CD2Z:
      ipStrides[0] = 1;
      ipStrides[1] = nx;
      opStrides[0] = 1;
      opStrides[1] = 1 + nx / 2;
      ipDistance = ny * nx;
      opDistance = ny * (1 + nx / 2);
      break;

    case HCFFT_C2RZ2D:
      ipStrides[0] = 1;
      ipStrides[1] = 1 + nx / 2;
      opStrides[0] = 1;
      opStrides[1] = nx;
      ipDistance = ny * (1 + nx / 2);
      opDistance = ny * nx;
      scale = 1.0;
      break;

    case HCFFT_C2CZ2Z:
      ipStrides[0] = 1;
      ipStrides[1] = nx;
      opStrides[0] = 1;
      opStrides[1] = nx;
      ipDistance = ny * nx;
      opDistance = ny * nx;
      break;

    default:
      // Invalid type
      return HCFFT_INVALID_VALUE;
  }

  hcfftStatus status = planObject.hcfftCreateDefaultPlan (plan, dimension, length, direction, precision, libType);

  if ( status == HCFFT_ERROR || status == HCFFT_INVALID ) {
    return HCFFT_INVALID_VALUE;
  }

  // Default options
  // set certain properties of plan with default values
  // Set Precision
  status = planObject.hcfftSetPlanPrecision(*plan, precision);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  // Set Transpose type
  status = planObject.hcfftSetPlanTransposeResult(*plan, HCFFT_NOTRANSPOSE);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  // Set Result location data layout
  status = planObject.hcfftSetResultLocation(*plan, HCFFT_OUTOFPLACE);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanInStride(*plan, dimension, ipStrides);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanOutStride(*plan, dimension, opStrides);

  if( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanDistance(*plan, ipDistance, opDistance);

  if( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  if ( libType == HCFFT_C2RZ2D) {
    status = planObject.hcfftSetPlanScale(*plan, direction, scale );

    if( status != HCFFT_SUCCEEDS) {
      return HCFFT_SETUP_FAILED;
    }
  }

  return HCFFT_SUCCESS;
}

/*
 * <iii> Function hcfftPlan3d()
   Description:
      Creates a 3D FFT plan configuration according to specified signal sizes and data type.
   This function is the same as hcfftPlan2d() except that it takes a third size parameter nz.

   Input:
   ----------------------------------------------------------------------------------------------
   #1 plan  Pointer to a hcfftHandle object
   #2 nx  The transform size in the x dimension
   #3 ny  The transform size in the y dimension
   #4 nz  The transform size in the z dimension
   #5 type  The transform data type (e.g., HCFFT_R2C for single precision real to complex)

   Output:
   ----------------------------------------------------------------------------------------------
   #1 plan  Contains a hcFFT 3D plan handle value

   Return Values:
   ----------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully created the FFT plan.
   HCFFT_ALLOC_FAILED   The allocation of GPU resources for the plan failed.
   HCFFT_INVALID_VALUE  One or more invalid parameters were passed to the API.
   HCFFT_INTERNAL_ERROR   An internal driver error was detected.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
   HCFFT_INVALID_SIZE   One or more of the nx, ny, or nz parameters is not a supported size.
*/

hcfftResult hcfftPlan3d(hcfftHandle*&plan, int nx, int ny, int nz, hcfftType type) {
  // Set dimension as 3D
  hcfftDim dimension = HCFFT_3D;
  // Check the input type and set appropriate direction and precision
  hcfftDirection direction;
  hcfftPrecision precision;

  switch (type) {
    case HCFFT_R2C:
      precision = HCFFT_SINGLE;
      direction = HCFFT_FORWARD;
      break;

    case HCFFT_C2R:
      precision = HCFFT_SINGLE;
      direction = HCFFT_BACKWARD;
      break;

    case HCFFT_C2C:
      precision = HCFFT_SINGLE;
      direction = HCFFT_BOTH;
      break;

    case HCFFT_D2Z:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_FORWARD;
      break;

    case HCFFT_Z2D:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_BACKWARD;
      break;

    case HCFFT_Z2Z:
      precision = HCFFT_DOUBLE;
      direction = HCFFT_BOTH;
      break;

    default:
      // Invalid type
      return HCFFT_INVALID_VALUE;
  }

  // length array to bookkeep dimension info
  size_t* length = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* ipStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t* opStrides = (size_t*)malloc(sizeof(size_t) * dimension);
  size_t ipDistance, opDistance;
  float scale;

  if (nx < 0 || ny < 0 || nz < 0) {
    // invalid size
    return HCFFT_INVALID_SIZE;
  } else {
    length[0] = nx;
    length[1] = ny;
    length[2] = nz;
  }

  // Allocate Rawplan
  hcfftResult res = hcfftCreate(plan);

  if(res != HCFFT_SUCCESS) {
    return HCFFT_ALLOC_FAILED;
  }

  hc::accelerator acc;
  res = hcfftXtSetGPUs(acc);

  if(res != HCFFT_SUCCESS) {
    return HCFFT_SETUP_FAILED;
  }

  hcfftLibType libType = ((type == HCFFT_R2C || type == HCFFT_D2Z) ? HCFFT_R2CD2Z : (type == HCFFT_C2R || type == HCFFT_Z2D) ? HCFFT_C2RZ2D : (type == HCFFT_C2C || type == HCFFT_Z2Z ) ? HCFFT_C2CZ2Z : (hcfftLibType)0);

  switch (libType) {
    case HCFFT_R2CD2Z:
      ipStrides[0] = 1;
      ipStrides[1] = nx;
      ipStrides[2] = nx * ny;
      opStrides[0] = 1;
      opStrides[1] = 1 + (nx / 2);
      opStrides[2] = (1 + (nx / 2)) * ny;
      ipDistance = nz * ny * nx;
      opDistance = nz * ny * (1 + nx / 2);
      break;

    case HCFFT_C2RZ2D:
      ipStrides[0] = 1;
      ipStrides[1] = 1 + (nx / 2);
      ipStrides[2] = (1 + (nx / 2)) * ny;
      opStrides[0] = 1;
      opStrides[1] = nx;
      opStrides[2] = nx * ny;
      ipDistance = nz * ny * (1 + nx / 2);
      opDistance = nz * ny * nx;
      scale = 1.0;
      break;

    case HCFFT_C2CZ2Z:
      ipStrides[0] = 1;
      ipStrides[1] = nx;
      ipStrides[2] = nx * ny;
      opStrides[0] = 1;
      opStrides[1] = nx;
      opStrides[2] = nx * ny;
      ipDistance = nz * ny * nx;
      opDistance = nz * ny * nx;
      break;

    default:
      // Invalid type
      return HCFFT_INVALID_VALUE;
  }

  hcfftStatus status = planObject.hcfftCreateDefaultPlan (plan, dimension, length, direction, precision, libType);

  if ( status == HCFFT_ERROR || status == HCFFT_INVALID ) {
    return HCFFT_INVALID_VALUE;
  }

  // Default options
  // set certain properties of plan with default values
  // Set Precision
  status = planObject.hcfftSetPlanPrecision(*plan, precision);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  // Set Transpose type
  status = planObject.hcfftSetPlanTransposeResult(*plan, HCFFT_NOTRANSPOSE);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  // Set Result location data layout
  status = planObject.hcfftSetResultLocation(*plan, HCFFT_OUTOFPLACE);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanInStride(*plan, dimension, ipStrides);

  if ( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanOutStride(*plan, dimension, opStrides);

  if( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetPlanDistance(*plan, ipDistance, opDistance);

  if( status != HCFFT_SUCCEEDS ) {
    return HCFFT_SETUP_FAILED;
  }

  if ( libType == HCFFT_C2RZ2D) {
    status = planObject.hcfftSetPlanScale(*plan, direction, scale );

    if( status != HCFFT_SUCCEEDS) {
      return HCFFT_SETUP_FAILED;
    }
  }

  return HCFFT_SUCCESS;
}

/* Function hcfftDestroy()
   Description:
      Frees all GPU resources associated with a hcFFT plan and destroys the internal plan data structure.
   This function should be called once a plan is no longer needed, to avoid wasting GPU memory.

   Input:
   -----------------------------------------------------------------------------------------------------
   plan         The hcfftHandle object of the plan to be destroyed.

   Return Values:
   -----------------------------------------------------------------------------------------------------
   HCFFT_SUCCESS        hcFFT successfully destroyed the FFT plan.
   HCFFT_INVALID_PLAN   The plan parameter is not a valid handle.
*/

hcfftResult hcfftDestroy(hcfftHandle plan) {
  auto planHandle = plan;
  hcfftStatus status = planObject.hcfftDestroyPlan(&planHandle);

  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_INVALID_PLAN;
  }

  return HCFFT_SUCCESS;
}

/* Functions hcfftExecR2C() and hcfftExecD2Z()
   Description:
      hcfftExecR2C() (hcfftExecD2Z()) executes a single-precision (double-precision) real-to-complex, implicitly forward,
   hcFFT transform plan. hcFFT uses as input data the GPU memory pointed to by the idata parameter. This function stores
   the nonredundant Fourier coefficients in the odata array. Pointers to idata and odata are both required to be aligned
   to hcfftComplex data type in single-precision transforms and hcfftDoubleComplex data type in double-precision transforms.
   If idata and odata are the same, this method does an in-place transform. Note the data layout differences between in-place
   and out-of-place transforms as described in Parameter hcfftType.

   Input:
   -----------------------------------------------------------------------------------------------------------------------
   plan   hcfftHandle returned by hcfftCreate
   idata  Pointer to the real input data (in GPU memory) to transform
   odata  Pointer to the complex output data (in GPU memory)

   Output:
   -----------------------------------------------------------------------------------------------------------------------
   odata  Contains the complex Fourier coefficients

   Return Values:
   ------------------------------------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully executed the FFT plan.
   HCFFT_INVALID_PLAN   The plan parameter is not a valid handle.
   HCFFT_INVALID_VALUE  At least one of the parameters idata and odata is not valid.
   HCFFT_INTERNAL_ERROR   An internal driver error was detected.
   HCFFT_EXEC_FAILED  hcFFT failed to execute the transform on the GPU.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
*/

hcfftResult hcfftExecR2C(hcfftHandle plan, hcfftReal* idata, hcfftComplex* odata) {
  // Nullity check
  if( idata == NULL || odata == NULL) {
    return HCFFT_INVALID_VALUE;
  }

  // TODO: Check validity of plan
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftReal* odataR = (hcfftReal*)odata;
  hcfftResLocation loc = HCFFT_OUTOFPLACE;

  if(idata == odataR) {
    loc = HCFFT_INPLACE;
  }

  hcfftStatus status = planObject.hcfftSetLayout(plan, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetResultLocation(plan, loc);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftBakePlan(plan);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftEnqueueTransform(plan, dir, idata, odataR, NULL);

  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_EXEC_FAILED;
  }

  return HCFFT_SUCCESS;
}

hcfftResult hcfftExecD2Z(hcfftHandle plan, hcfftDoubleReal* idata, hcfftDoubleComplex* odata) {
  // Nullity check
  if( idata == NULL || odata == NULL) {
    return HCFFT_INVALID_VALUE;
  }

  // TODO: Check validity of plan
  hcfftDirection dir = HCFFT_FORWARD;
  hcfftDoubleReal* odataR = (hcfftDoubleReal*)odata;
  hcfftResLocation loc = HCFFT_OUTOFPLACE;

  if(idata == odataR) {
    loc = HCFFT_INPLACE;
  }

  hcfftStatus status = planObject.hcfftSetLayout(plan, HCFFT_REAL, HCFFT_HERMITIAN_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetResultLocation(plan, loc);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftBakePlan(plan);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftEnqueueTransform(plan, dir, idata, odataR, NULL);

  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_EXEC_FAILED;
  }

  return HCFFT_SUCCESS;
}

/* Functions hcfftExecC2R() and hcfftExecZ2D()
   Description:
     hcfftExecC2R() (hcfftExecZ2D()) executes a single-precision (double-precision) complex-to-real,
   implicitly inverse, hcFFT transform plan. hcFFT uses as input data the GPU memory pointed to by the
   idata parameter. The input array holds only the nonredundant complex Fourier coefficients. This function
   stores the real output values in the odata array. and pointers are both required to be aligned to hcfftComplex
   data type in single-precision transforms and hcfftDoubleComplex type in double-precision transforms. If idata
   and odata are the same, this method does an in-place transform.

   Input:
   ------------------------------------------------------------------------------------------------------------------
   plan   hcfftHandle returned by hcfftCreate
   idata  Pointer to the complex input data (in GPU memory) to transform
   odata  Pointer to the real output data (in GPU memory)

   Output:
   ------------------------------------------------------------------------------------------------------------------
   odata  Contains the real output data

   Return Values:
   -------------------------------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully executed the FFT plan.
   HCFFT_INVALID_PLAN   The plan parameter is not a valid handle.
   HCFFT_INVALID_VALUE  At least one of the parameters idata and odata is not valid.
   HCFFT_INTERNAL_ERROR   An internal driver error was detected.
   HCFFT_EXEC_FAILED  hcFFT failed to execute the transform on the GPU.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize.
*/

hcfftResult hcfftExecC2R(hcfftHandle plan, hcfftComplex* idata, hcfftReal* odata) {
  // Nullity check
  if( idata == NULL || odata == NULL) {
    return HCFFT_INVALID_VALUE;
  }

  // TODO: Check validity of plan
  hcfftDirection dir = HCFFT_BACKWARD;
  hcfftReal* idataR = (hcfftReal*)idata;
  hcfftResLocation loc = HCFFT_OUTOFPLACE;

  if(idataR == odata) {
    loc = HCFFT_INPLACE;
  }

  hcfftStatus status = planObject.hcfftSetLayout(plan, HCFFT_HERMITIAN_INTERLEAVED, HCFFT_REAL);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetResultLocation(plan, loc);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftBakePlan(plan);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftEnqueueTransform(plan, dir, idataR, odata, NULL);

  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_EXEC_FAILED;
  }

  return HCFFT_SUCCESS;
}

hcfftResult hcfftExecZ2D(hcfftHandle plan, hcfftDoubleComplex* idata, hcfftDoubleReal* odata) {
  // Nullity check
  if( idata == NULL || odata == NULL) {
    return HCFFT_INVALID_VALUE;
  }

  // TODO: Check validity of plan
  hcfftDirection dir = HCFFT_BACKWARD;
  hcfftDoubleReal* idataR = (hcfftDoubleReal*)idata;
  hcfftResLocation loc = HCFFT_OUTOFPLACE;

  if(idataR == odata) {
    loc = HCFFT_INPLACE;
  }

  hcfftStatus status = planObject.hcfftSetLayout(plan, HCFFT_HERMITIAN_INTERLEAVED, HCFFT_REAL);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetResultLocation(plan, loc);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftBakePlan(plan);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftEnqueueTransform(plan, dir, idataR, odata, NULL);

  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_EXEC_FAILED;
  }

  return HCFFT_SUCCESS;
}

/* Functions hcfftExecC2C() and hcfftExecZ2Z()
   Description:
     hcfftExecC2C() (hcfftExecZ2Z()) executes a single-precision (double-precision) complex-to-complex transform
   plan in the transform direction as specified by direction parameter. hcFFT uses the GPU memory pointed to by the
   idata parameter as input data. This function stores the Fourier coefficients in the odata array.
   If idata and odata are the same, this method does an in-place transform.

   Input:
   ----------------------------------------------------------------------------------------------------------
   plan   hcfftHandle returned by hcfftCreate
   idata  Pointer to the complex input data (in GPU memory) to transform
   odata  Pointer to the complex output data (in GPU memory)
   direction  The transform direction: HCFFT_FORWARD or HCFFT_INVERSE

   Output:
   -----------------------------------------------------------------------------------------------------------
   odata  Contains the complex Fourier coefficients

   Return Values:
   ------------------------------------------------------------------------------------------------------------
   HCFFT_SUCCESS  hcFFT successfully executed the FFT plan.
   HCFFT_INVALID_PLAN   The plan parameter is not a valid handle.
   HCFFT_INVALID_VALUE  At least one of the parameters idata, odata, and direction is not valid.
   HCFFT_INTERNAL_ERROR   An internal driver error was detected.
   HCFFT_EXEC_FAILED  hcFFT failed to execute the transform on the GPU.
   HCFFT_SETUP_FAILED   The hcFFT library failed to initialize. */

hcfftResult hcfftExecC2C(hcfftHandle plan, hcfftComplex* idata, hcfftComplex* odata, int direction) {
  // Nullity check
  if( idata == NULL || odata == NULL) {
    return HCFFT_INVALID_VALUE;
  }

  // TODO: Check validity of plan
  hcfftReal* idataR = (hcfftReal*)idata;
  hcfftReal* odataR = (hcfftReal*)odata;
  hcfftResLocation loc = HCFFT_OUTOFPLACE;

  if(idataR == odataR) {
    loc = HCFFT_INPLACE;
  }

  hcfftStatus status = planObject.hcfftSetLayout(plan, HCFFT_COMPLEX_INTERLEAVED, HCFFT_COMPLEX_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetResultLocation(plan, loc);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftBakePlan(plan);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftEnqueueTransform(plan, (hcfftDirection)direction, idataR, odataR, NULL);

  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_EXEC_FAILED;
  }

  return HCFFT_SUCCESS;
}

hcfftResult hcfftExecZ2Z(hcfftHandle plan, hcfftDoubleComplex* idata, hcfftDoubleComplex* odata, int direction) {
  // Nullity check
  if( idata == NULL || odata == NULL) {
    return HCFFT_INVALID_VALUE;
  }

  // TODO: Check validity of plan
  hcfftDoubleReal* idataR = (hcfftDoubleReal*)idata;
  hcfftDoubleReal* odataR = (hcfftDoubleReal*)odata;
  hcfftResLocation loc = HCFFT_OUTOFPLACE;

  if(idataR == odataR) {
    loc = HCFFT_INPLACE;
  }

  hcfftStatus status = planObject.hcfftSetLayout(plan, HCFFT_COMPLEX_INTERLEAVED, HCFFT_COMPLEX_INTERLEAVED);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftSetResultLocation(plan, loc);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftBakePlan(plan);

  if(status != HCFFT_SUCCEEDS) {
    return HCFFT_SETUP_FAILED;
  }

  status = planObject.hcfftEnqueueTransform(plan, (hcfftDirection)direction, idataR, odataR, NULL);

  if (status != HCFFT_SUCCEEDS) {
    return HCFFT_EXEC_FAILED;
  }

  return HCFFT_SUCCESS;
}
