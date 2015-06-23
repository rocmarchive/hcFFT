#include "ampfftlib.h"

/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/

//	This routine will query the OpenCL context for it's devices
//	and their hardware limitations, which we synthesize into a
//	hardware "envelope".
//	We only query the devices the first time we're called after
//	the object's context is set.  On 2nd and subsequent calls,
//	we just return the pointer.
//
ampfftStatus FFTPlan::SetEnvelope ()
{

	// TODO  The caller has already acquired the lock on *this
	//	However, we shouldn't depend on it.

        envelope.limit_LocalMemSize = 32768;
        envelope.limit_WorkGroupSize = 256;
        envelope.limit_Dimensions = 3;
        for(int i = 0 ;i < envelope.limit_Dimensions; i++)
          envelope.limit_Size[i] = 256;

	return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::GetEnvelope (const FFTEnvelope ** ppEnvelope) const
{
	if(&envelope == NULL) assert(false);
	*ppEnvelope = &envelope;
	return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftCreateDefaultPlan (ampfftPlanHandle* plHandle,ampfftDim dimension, const size_t *length)
{
  if( length == NULL )
    return AMPFFT_ERROR;

  size_t lenX = 1, lenY = 1, lenZ = 1;
  switch( dimension )
  {
    case AMPFFT_1D:
    {
      if( length[ 0 ] == 0 )
	return AMPFFT_ERROR;
      lenX = length[ 0 ];
    }
    break;

    case AMPFFT_2D:
    {
      if( length[ 0 ] == 0 || length[ 1 ] == 0 )
	return AMPFFT_ERROR;
      lenX = length[ 0 ];
      lenY = length[ 1 ];
    }
    break;

    case AMPFFT_3D:
    {
      if( length[ 0 ] == 0 || length[ 1 ] == 0 || length[ 2 ] == 0 )
	return AMPFFT_ERROR;
      lenX = length[ 0 ];
      lenY = length[ 1 ];
      lenZ = length[ 2 ];
    }
    break;

    default:
      return AMPFFT_ERROR;
    break;
  }

  FFTPlan* fftPlan = NULL;
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  fftRepo.createPlan( plHandle, fftPlan );
  fftPlan->baked = false;
  fftPlan->dimension = dimension;
  fftPlan->location = AMPFFT_INPLACE;
  fftPlan->ipLayout = AMPFFT_COMPLEX;
  fftPlan->opLayout = AMPFFT_COMPLEX;
  fftPlan->precision = AMPFFT_SINGLE;
  fftPlan->forwardScale	= 1.0;
  fftPlan->backwardScale = 1.0 / static_cast< double >( lenX * lenY * lenZ );
  fftPlan->batchSize = 1;

  fftPlan->gen = Stockham; //default setting

  fftPlan->SetEnvelope();

  switch( dimension )
  {
    case AMPFFT_1D:
    {
      fftPlan->length.push_back( lenX );
      fftPlan->inStride.push_back( 1 );
      fftPlan->outStride.push_back( 1 );
      fftPlan->iDist		= lenX;
      fftPlan->oDist		= lenX;
    }
    break;

    case AMPFFT_2D:
    {
      fftPlan->length.push_back( lenX );
      fftPlan->length.push_back( lenY );
      fftPlan->inStride.push_back( 1 );
      fftPlan->inStride.push_back( lenX );
      fftPlan->outStride.push_back( 1 );
      fftPlan->outStride.push_back( lenX );
      fftPlan->iDist		= lenX*lenY;
      fftPlan->oDist		= lenX*lenY;
    }
    break;

    case AMPFFT_3D:
    {
      fftPlan->length.push_back( lenX );
      fftPlan->length.push_back( lenY );
      fftPlan->length.push_back( lenZ );
      fftPlan->inStride.push_back( 1 );
      fftPlan->inStride.push_back( lenX );
      fftPlan->inStride.push_back( lenX*lenY );
      fftPlan->outStride.push_back( 1 );
      fftPlan->outStride.push_back( lenX );
      fftPlan->outStride.push_back( lenX*lenY );
      fftPlan->iDist		= lenX*lenY*lenZ;
      fftPlan->oDist		= lenX*lenY*lenZ;
    }
    break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::executePlan(FFTPlan* fftPlan)
{
  if(!fftPlan)
    return AMPFFT_INVALID;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanPrecision( const  ampfftPlanHandle plHandle,  ampfftPrecision* precision )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftGetPlanPrecision" ) );

  *precision = fftPlan->precision;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanPrecision(  ampfftPlanHandle plHandle,  ampfftPrecision precision )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftSetPlanPrecision" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->precision = precision;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanScale( const  ampfftPlanHandle plHandle,  ampfftDirection dir, float* scale )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftGetPlanScale" ) );

  if( dir == AMPFFT_FORWARD)
    *scale = (float)(fftPlan->forwardScale);
  else
    *scale = (float)(fftPlan->backwardScale);

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanScale(  ampfftPlanHandle plHandle,  ampfftDirection dir, float scale )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftSetPlanScale" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  if( dir == AMPFFT_FORWARD)
    fftPlan->forwardScale = scale;
  else
    fftPlan->backwardScale = scale;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanBatchSize( const  ampfftPlanHandle plHandle, size_t* batchsize )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan(plHandle, fftPlan, planLock );
  scopedLock sLock(*planLock, _T( " ampfftGetPlanBatchSize" ) );

  *batchsize = fftPlan->batchSize;
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanBatchSize( ampfftPlanHandle plHandle, size_t batchsize )
{
 FFTRepo& fftRepo = FFTRepo::getInstance( );
 FFTPlan* fftPlan = NULL;
 lockRAII* planLock = NULL;

 fftRepo.getPlan(plHandle, fftPlan, planLock );
 scopedLock sLock(*planLock, _T( " ampfftSetPlanBatchSize" ) );

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->batchSize = batchsize;
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanDim( const ampfftPlanHandle plHandle,  ampfftDim* dim, int* size )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanDim" ) );

  *dim = fftPlan->dimension;

  switch( fftPlan->dimension )
  {
    case AMPFFT_1D:
    {
      *size = 1;
    }
    break;
    case AMPFFT_2D:
    {
      *size = 2;
    }
    break;
    case AMPFFT_3D:
    {
      *size = 3;
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanDim(  ampfftPlanHandle plHandle, const  ampfftDim dim )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanDim" ) );

  // We resize the vectors in the plan to keep their sizes consistent with the value of the dimension
  switch( dim )
  {
    case AMPFFT_1D:
    {
      fftPlan->length.resize( 1 );
      fftPlan->inStride.resize( 1 );
      fftPlan->outStride.resize( 1 );
    }
    break;
    case AMPFFT_2D:
    {
      fftPlan->length.resize( 2 );
      fftPlan->inStride.resize( 2 );
      fftPlan->outStride.resize( 2 );
    }
    break;
    case AMPFFT_3D:
    {
      fftPlan->length.resize( 3 );
      fftPlan->inStride.resize( 3 );
      fftPlan->outStride.resize( 3 );
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;
  fftPlan->dimension = dim;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanLength( const  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clLengths )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanLength" ) );

  if( clLengths == NULL )
    return AMPFFT_ERROR;

  if( fftPlan->length.empty( ) )
    return AMPFFT_ERROR;

  switch( dim )
  {
    case AMPFFT_1D:
    {
      clLengths[0] = fftPlan->length[0];
    }
    break;
    case AMPFFT_2D:
    {
      if( fftPlan->length.size() < 2 )
        return AMPFFT_ERROR;

      clLengths[0] = fftPlan->length[0];
      clLengths[1 ] = fftPlan->length[1];
    }
    break;
    case AMPFFT_3D:
    {
      if( fftPlan->length.size() < 3 )
	return AMPFFT_ERROR;

      clLengths[0] = fftPlan->length[0];
      clLengths[1 ] = fftPlan->length[1];
      clLengths[2] = fftPlan->length[2];
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanLength(  ampfftPlanHandle plHandle, const  ampfftDim dim, const size_t* clLengths )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetPlanLength" ) );

  if( clLengths == NULL )
    return AMPFFT_ERROR;

  //	Simplest to clear any previous contents, because it's valid for user to shrink dimension
  fftPlan->length.clear( );
  switch( dim )
  {
    case AMPFFT_1D:
    {
    //	Minimum length size is 1
    if( clLengths[0] == 0 )
      return AMPFFT_ERROR;

    fftPlan->length.push_back( clLengths[0] );
    }
    break;
    case AMPFFT_2D:
    {
      //	Minimum length size is 1
      if(clLengths[0] == 0 || clLengths[1] == 0 )
         return AMPFFT_ERROR;

      fftPlan->length.push_back( clLengths[0] );
      fftPlan->length.push_back( clLengths[1] );
    }
    break;
    case AMPFFT_3D:
    {
      //	Minimum length size is 1
      if(clLengths[0 ] == 0 || clLengths[1] == 0 || clLengths[2] == 0)
        return AMPFFT_ERROR;

      fftPlan->length.push_back( clLengths[0] );
      fftPlan->length.push_back( clLengths[1] );
      fftPlan->length.push_back( clLengths[2] );
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  fftPlan->dimension = dim;

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftGetPlanInStride( const  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clStrides )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftGetPlanInStride" ) );

  if(clStrides == NULL )
    return AMPFFT_ERROR;

  switch( dim )
  {
    case AMPFFT_1D:
    {
      if(fftPlan->inStride.size( ) > 0 )
        clStrides[0] = fftPlan->inStride[0];
      else
        return AMPFFT_ERROR;
    }
    break;
    case AMPFFT_2D:
    {
      if( fftPlan->inStride.size( ) > 1 )
      {
        clStrides[0] = fftPlan->inStride[0];
	clStrides[1] = fftPlan->inStride[1];
      }
      else
        return AMPFFT_ERROR;
    }
    break;
    case AMPFFT_3D:
    {
      if( fftPlan->inStride.size( ) > 2 )
      {
        clStrides[0] = fftPlan->inStride[0];
        clStrides[1] = fftPlan->inStride[1];
	clStrides[2] = fftPlan->inStride[2];
      }
      else
        return AMPFFT_ERROR;
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftSetPlanInStride(  ampfftPlanHandle plHandle, const  ampfftDim dim, size_t* clStrides )
{
  FFTRepo& fftRepo = FFTRepo::getInstance( );
  FFTPlan* fftPlan = NULL;
  lockRAII* planLock = NULL;

  fftRepo.getPlan( plHandle, fftPlan, planLock );
  scopedLock sLock( *planLock, _T( " ampfftSetPlanInStride" ) );

  if( clStrides == NULL )
    return AMPFFT_ERROR;

  //	Simplest to clear any previous contents, because it's valid for user to shrink dimension
  fftPlan->inStride.clear( );
  switch( dim )
  {
    case AMPFFT_1D:
    {
      fftPlan->inStride.push_back( clStrides[0] );
    }
    break;
    case AMPFFT_2D:
    {
      fftPlan->inStride.push_back( clStrides[0] );
      fftPlan->inStride.push_back( clStrides[1] );
    }
    break;
    case AMPFFT_3D:
    {
      fftPlan->inStride.push_back( clStrides[0] );
      fftPlan->inStride.push_back( clStrides[1] );
      fftPlan->inStride.push_back( clStrides[2] );
    }
    break;
    default:
      return AMPFFT_ERROR;
      break;
  }

  //	If we modify the state of the plan, we assume that we can't trust any pre-calculated contents anymore
  fftPlan->baked = false;

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ampfftDestroyPlan( ampfftPlanHandle* plHandle )
{
  FFTRepo& fftRepo	= FFTRepo::getInstance( );
  FFTPlan* fftPlan	= NULL;
  lockRAII* planLock	= NULL;

  fftRepo.getPlan( *plHandle, fftPlan, planLock );

  //	Recursively destroy subplans, that are used for higher dimensional FFT's
  if( fftPlan->planX )
    ampfftDestroyPlan( &fftPlan->planX );

  if( fftPlan->planY )
    ampfftDestroyPlan( &fftPlan->planY );

   if( fftPlan->planZ )
    ampfftDestroyPlan( &fftPlan->planZ );

   if( fftPlan->planTX )
    ampfftDestroyPlan( &fftPlan->planTX );

   if( fftPlan->planTY )
    ampfftDestroyPlan( &fftPlan->planTY );

   if( fftPlan->planTZ )
    ampfftDestroyPlan( &fftPlan->planTZ );

   if( fftPlan->planRCcopy )
    ampfftDestroyPlan( &fftPlan->planRCcopy );

  fftRepo.deletePlan( plHandle );

  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::AllocateWriteBuffers ()
{
	ampfftStatus status = AMPFFT_SUCCESS;

	assert (NULL == const_buffer);

	assert(4 == sizeof(int));

	//	Construct the constant buffer and call clEnqueueWriteBuffer
	float ConstantBufferParams[AMPFFT_CB_SIZE];
	memset (& ConstantBufferParams, 0, sizeof (ConstantBufferParams));

	float nY = 1;
	float nZ = 0;
	float nW = 0;
	float n5 = 0;

	switch( length.size() )
	{
	case 1:
		nY = (float)batchSize;
		break;

	case 2:
		nY = (float)length[1];
		nZ = (float)batchSize;
		break;

	case 3:
		nY = (float)length[1];
		nZ = (float)length[2];
		nW = (float)batchSize;
		break;

	case 4:
		nY = (float)length[1];
		nZ = (float)length[2];
		nW = (float)length[3];
		n5 = (float)batchSize;
		break;
	}
	ConstantBufferParams[AMPFFT_CB_NY ] = nY;
	ConstantBufferParams[AMPFFT_CB_NZ ] = nZ;
	ConstantBufferParams[AMPFFT_CB_NW ] = nW;
	ConstantBufferParams[AMPFFT_CB_N5 ] = n5;

	assert (/*fftPlan->*/inStride.size() == /*fftPlan->*/outStride.size());

        std::cout<<" inStride.size() "<<inStride.size() << " outStride.size() "<<outStride.size() <<std::endl;
	switch (/*fftPlan->*/inStride.size()) {
	case 1:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)iDist;
		break;

	case 2:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)inStride[1];
		ConstantBufferParams[AMPFFT_CB_ISZ] = (float)iDist;
		break;

	case 3:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)inStride[1];
		ConstantBufferParams[AMPFFT_CB_ISZ] = (float)inStride[2];
		ConstantBufferParams[AMPFFT_CB_ISW] = (float)iDist;
		break;

	case 4:
		ConstantBufferParams[AMPFFT_CB_ISX] = (float)inStride[0];
		ConstantBufferParams[AMPFFT_CB_ISY] = (float)inStride[1];
		ConstantBufferParams[AMPFFT_CB_ISZ] = (float)inStride[2];
		ConstantBufferParams[AMPFFT_CB_ISW] = (float)inStride[3];
		ConstantBufferParams[AMPFFT_CB_IS5] = (float)iDist;
		break;
	}

	switch (/*fftPlan->*/outStride.size()) {
	case 1:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)oDist;
		break;

	case 2:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)outStride[1];
		ConstantBufferParams[AMPFFT_CB_OSZ] = (float)oDist;
		break;

	case 3:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)outStride[1];
		ConstantBufferParams[AMPFFT_CB_OSZ] = (float)outStride[2];
		ConstantBufferParams[AMPFFT_CB_OSW] = (float)oDist;
		break;

	case 4:
		ConstantBufferParams[AMPFFT_CB_OSX] = (float)outStride[0];
		ConstantBufferParams[AMPFFT_CB_OSY] = (float)outStride[1];
		ConstantBufferParams[AMPFFT_CB_OSZ] = (float)outStride[2];
		ConstantBufferParams[AMPFFT_CB_OSW] = (float)outStride[3];
		ConstantBufferParams[AMPFFT_CB_OS5] = (float)oDist;
		break;
	}

        Concurrency::array<float, 1> arr = Concurrency::array<float, 1>(Concurrency::extent<1>(AMPFFT_CB_SIZE), ConstantBufferParams);
        const_buffer = new Concurrency::array_view<float>(arr);
        for(int i = 0 ; i < 32 ; i++)
          std::cout<<" const_buffer["<<i<<"] "<<(*const_buffer)[i]<<std::endl;
	return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::ReleaseBuffers ()
{
	ampfftStatus result = AMPFFT_SUCCESS;

	if( NULL != const_buffer )
	{
                delete const_buffer;
	}

	if( NULL != intBuffer )
	{
                delete intBuffer;
	}

	if( NULL != intBufferRC )
	{
                delete intBufferRC;
	}

	return	AMPFFT_SUCCESS;
}

size_t FFTPlan::ElementSize() const
{
  return ((precision == AMPFFT_DOUBLE) ? sizeof(std::complex<double> ) : sizeof(std::complex<float>));
}
/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/

/*---------------------------------------------------FFTRepo--------------------------------------------------------------------------------*/
ampfftStatus FFTRepo::createPlan( ampfftPlanHandle* plHandle, FFTPlan*& fftPlan )
{
	scopedLock sLock( lockRepo, _T( "insertPlan" ) );

	//	We keep track of this memory in our own collection class, to make sure it's freed in releaseResources
	//	The lifetime of a plan is tracked by the client and is freed when the client calls ::ampfftDestroyPlan()
	fftPlan	= new FFTPlan;

	//	We allocate a new lock here, and expect it to be freed in ::ampfftDestroyPlan();
	//	The lifetime of the lock is the same as the lifetime of the plan
	lockRAII* lockPlan	= new lockRAII;

	//	Add and remember the fftPlan in our map
	repoPlans[ planCount ] = make_pair( fftPlan, lockPlan );

	//	Assign the user handle the plan count (unique identifier), and bump the count for the next plan
	*plHandle	= planCount++;

	return	AMPFFT_SUCCESS;
}


ampfftStatus FFTRepo::getPlan( ampfftPlanHandle plHandle, FFTPlan*& fftPlan, lockRAII*& planLock )
{
	scopedLock sLock( lockRepo, _T( "getPlan" ) );

	//	First, check if we have already created a plan with this exact same FFTPlan
	repoPlansType::iterator iter	= repoPlans.find( plHandle );
	if( iter == repoPlans.end( ) )
		return	AMPFFT_ERROR;

	//	If plan is valid, return fill out the output pointers
	fftPlan		= iter->second.first;
	planLock	= iter->second.second;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::deletePlan( ampfftPlanHandle* plHandle )
{
	scopedLock sLock( lockRepo, _T( "deletePlan" ) );

	//	First, check if we have already created a plan with this exact same FFTPlan
	repoPlansType::iterator iter	= repoPlans.find( *plHandle );
	if( iter == repoPlans.end( ) )
		return	AMPFFT_ERROR;

	//	We lock the plan object while we are in the process of deleting it
	{
		scopedLock sLock( *iter->second.second, _T( "ampfftDestroyPlan" ) );

		//	Delete the FFTPlan
		delete iter->second.first;
	}

		//	Delete the lockRAII
	delete iter->second.second;

	//	Remove entry from our map object
	repoPlans.erase( iter );

	//	Clear the client's handle to signify that the plan is gone
	*plHandle = 0;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::setProgramEntryPoints( const ampfftGenerators gen, const ampfftPlanHandle& handle,
                                             const FFTKernelGenKeyParams& fftParam, const char * kernel_fwd,
                                             const char * kernel_back)
{
	scopedLock sLock( lockRepo, _T( "setProgramEntryPoints" ) );

	fftRepoKey key = std::make_pair( gen, handle );

	fftRepoValue& fft = mapFFTs[ key ];
	fft.EntryPoint_fwd  = kernel_fwd;
	fft.EntryPoint_back = kernel_back;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::getProgramEntryPoint( const ampfftGenerators gen, const ampfftPlanHandle& handle,
                                            const FFTKernelGenKeyParams& fftParam, ampfftDirection dir,
                                            std::string& kernel)
{
	scopedLock sLock( lockRepo, _T( "getProgramEntryPoint" ) );

	fftRepoKey key = std::make_pair( gen, handle );

	fftRepo_iterator pos = mapFFTs.find( key );

	if( pos == mapFFTs.end( ) )
		return	AMPFFT_ERROR;

	switch (dir) {
	case AMPFFT_FORWARD:
		kernel = pos->second.EntryPoint_fwd;
		break;
	case AMPFFT_BACKWARD:
		kernel = pos->second.EntryPoint_back;
		break;
	default:
		assert (false);
		return AMPFFT_ERROR;
	}

	if (0 == kernel.size())
		return	AMPFFT_ERROR;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::setProgramCode( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, const std::string& kernel)
{
	scopedLock sLock( lockRepo, _T( "setProgramCode" ) );

	fftRepoKey key = std::make_pair( gen, handle );

	// Prefix copyright statement at the top of generated kernels
	std::stringstream ss;
	ss <<
		"/* ************************************************************************\n"
		" * Copyright 2013 MCW, Inc.\n"
		" *\n"
		" * ************************************************************************/"
	<< std::endl << std::endl;

	std::string prefixCopyright = ss.str();

	mapFFTs[ key ].ProgramString = prefixCopyright + kernel;

	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::getProgramCode( const ampfftGenerators gen, const ampfftPlanHandle& handle, const FFTKernelGenKeyParams& fftParam, std::string& kernel)
{

	scopedLock sLock( lockRepo, _T( "getProgramCode" ) );
	fftRepoKey key = std::make_pair( gen, handle );

	fftRepo_iterator pos = mapFFTs.find( key);
	if( pos == mapFFTs.end( ) )
		return	AMPFFT_ERROR;

        kernel = pos->second.ProgramString;
	return	AMPFFT_SUCCESS;
}

ampfftStatus FFTRepo::releaseResources( )
{
	scopedLock sLock( lockRepo, _T( "releaseResources" ) );

	//	Free all memory allocated in the repoPlans; represents cached plans that were not destroyed by the client
	//
	for( repoPlansType::iterator iter = repoPlans.begin( ); iter != repoPlans.end( ); ++iter )
	{
		FFTPlan* plan	= iter->second.first;
		lockRAII* lock	= iter->second.second;
		if( plan != NULL )
		{
			delete plan;
		}
		if( lock != NULL )
		{
			delete lock;
		}
	}

	//	Reset the plan count to zero because we are guaranteed to have destroyed all plans
	planCount	= 1;

	//	Release all strings
	mapFFTs.clear( );

	return	AMPFFT_SUCCESS;
}
/*------------------------------------------------FFTRepo----------------------------------------------------------------------------------*/
