#include "ampfftlib.h"

/*----------------------------------------------------FFTPlan-----------------------------------------------------------------------------*/
ampfftStatus FFTPlan::ampfftCreateDefaultPlan (ampfftPlanHandle* plHandle,ampfftDim dimension, const size_t *length)
{
  return AMPFFT_SUCCESS;
}

ampfftStatus FFTPlan::executePlan(FFTPlan* fftPlan)
{
  if(!fftPlan)
    return AMPFFT_INVALID;

  return AMPFFT_SUCCESS;
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
/*------------------------------------------------FFTRepo----------------------------------------------------------------------------------*/
