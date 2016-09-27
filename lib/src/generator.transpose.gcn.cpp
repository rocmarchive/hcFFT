#include "generator.transpose.h"

// A structure that represents a bounding box or tile, with convenient names for the row and column addresses
// local work sizes
struct tile
{
    union
    {
        size_t x;
        size_t col;
    };

    union
    {
        size_t y;
        size_t row;
    };
};

static void OffsetCalc(std::stringstream& transKernel, const FFTKernelGenKeyParams& params, bool input )
{
	const size_t *stride = input ? params.fft_inStride : params.fft_outStride;
	std::string offset = input ? "iOffset" : "oOffset";


	hcKernWrite( transKernel, 3 ) << "size_t " << offset << " = 0;" << std::endl;
	hcKernWrite( transKernel, 3 ) << "currDimIndex = groupIndex.y;" << std::endl;


	for(size_t i = params.fft_DataDim - 2; i > 0 ; i--)
	{
		hcKernWrite( transKernel, 3 ) << offset << " += (currDimIndex/numGroupsY_" << i << ")*" << stride[i+1] << ";" << std::endl;
		hcKernWrite( transKernel, 3 ) << "currDimIndex = currDimIndex % numGroupsY_" << i << ";" << std::endl;
	}

	hcKernWrite( transKernel, 3 ) << "rowSizeinUnits = " << stride[1] << ";" << std::endl;

	if(params.transOutHorizontal)
	{
		if(input)
		{	
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += currDimIndex * wgTileExtent.x;" << std::endl;  
		}
		else
		{
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.x * currDimIndex;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += groupIndex.x * wgTileExtent.y * wgUnroll;" << std::endl;
		}
	}
	else
	{
		if(input)
		{	
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += groupIndex.x * wgTileExtent.x;" << std::endl;
		}
		else
		{
			hcKernWrite( transKernel, 3 ) << offset << " += rowSizeinUnits * wgTileExtent.x * groupIndex.x;" << std::endl;
			hcKernWrite( transKernel, 3 ) << offset << " += currDimIndex * wgTileExtent.y * wgUnroll;" << std::endl;  
		}
	}

	hcKernWrite( transKernel, 3 ) << std::endl;
}

// Small snippet of code that multiplies the twiddle factors into the butterfiles.  It is only emitted if the plan tells
// the generator that it wants the twiddle factors generated inside of the transpose
static hcfftStatus genTwiddleMath(const hcfftPlanHandle plHandle, const FFTKernelGenKeyParams& params, std::stringstream& transKernel, const std::string& dtComplex, bool fwd )
{
    hcKernWrite( transKernel, 9 ) << dtComplex << " W = TW3step" << plHandle << "( (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd) " <<  ", " << TwTableLargeName() << ");" << std::endl;
    hcKernWrite( transKernel, 9 ) << dtComplex << " T;" << std::endl;

	if(fwd)
	{
		hcKernWrite( transKernel, 9 ) << "T.x = ( W.x * tmp.x ) - ( W.y * tmp.y );" << std::endl;
		hcKernWrite( transKernel, 9 ) << "T.y = ( W.y * tmp.x ) + ( W.x * tmp.y );" << std::endl;
	}
	else
	{
		hcKernWrite( transKernel, 9 ) << "T.x =  ( W.x * tmp.x ) + ( W.y * tmp.y );" << std::endl;
		hcKernWrite( transKernel, 9 ) << "T.y = -( W.y * tmp.x ) + ( W.x * tmp.y );" << std::endl;
	}

    hcKernWrite( transKernel, 9 ) << "tmp.x = T.x;" << std::endl;
    hcKernWrite( transKernel, 9 ) << "tmp.y = T.y;" << std::endl;

    return HCFFT_SUCCEEDS;
}

// These strings represent the names that are used as strKernel parameters
const std::string pmRealIn( "pmRealIn" );
const std::string pmImagIn( "pmImagIn" );
const std::string pmRealOut( "pmRealOut" );
const std::string pmImagOut( "pmImagOut" );
const std::string pmComplexIn( "pmComplexIn" );
const std::string pmComplexOut( "pmComplexOut" );

static hcfftStatus genTransposePrototype( FFTKernelGenKeyParams& params, const tile& lwSize, const std::string& dtPlanar, const std::string& dtComplex,
    const std::string &funcName, std::stringstream& transKernel, std::string& dtInput, std::string& dtOutput, bool genTwiddle) {
  uint arg = 0;
  // Declare and define the function
  hcKernWrite( transKernel, 0 ) << "extern \"C\"\n { void" << std::endl;
  hcKernWrite( transKernel, 0 ) << funcName << "(  std::map<int, void*> vectArr, uint batchSize, accelerator_view &acc_view, accelerator &acc) \n {";

  switch( params.fft_inputLayout ) {
    case HCFFT_COMPLEX_INTERLEAVED:
      dtInput = dtComplex;
      hcKernWrite( transKernel, 0 ) << dtInput << " *"<<pmComplexIn<<" = static_cast< " << dtInput << "*> (vectArr[" << arg++ << "]);";

      switch( params.fft_placeness ) {
        case HCFFT_INPLACE:
          dtOutput = dtComplex;
          break;

        case HCFFT_OUTOFPLACE:
          switch( params.fft_outputLayout ) {
            case HCFFT_COMPLEX_INTERLEAVED:
              dtOutput = dtComplex;
              hcKernWrite( transKernel, 0 ) << dtOutput << " *"<< pmComplexOut << " = static_cast< " << dtOutput << "*> (vectArr[" << arg++ << "]);";
              break;

            case HCFFT_COMPLEX_PLANAR:
              dtOutput = dtPlanar;
              hcKernWrite( transKernel, 0 ) << dtOutput << " * "<< pmRealOut << " = static_cast< " << dtOutput << "*> (vectArr[" << arg++ << "]);";
              hcKernWrite( transKernel, 0 ) << dtOutput << "* "<< pmImagOut << " = static_cast< " << dtOutput << "*> (vectArr[" << arg++ << "]);";
              break;

            case HCFFT_HERMITIAN_INTERLEAVED:
            case HCFFT_HERMITIAN_PLANAR:
            case HCFFT_REAL:
            default:
              return HCFFT_INVALID;
          }

          break;

        default:
          return HCFFT_INVALID;
      }

      break;

    case HCFFT_COMPLEX_PLANAR:
      dtInput = dtPlanar;
      hcKernWrite( transKernel, 0 ) << dtInput << " * " << pmRealIn << " = static_cast< " << dtInput << "*> (vectArr[" << arg++ << "]);";
      hcKernWrite( transKernel, 0 ) << dtInput << " * " << pmImagIn << " = static_cast< " << dtInput << "*> (vectArr[" << arg++ << "]);";

      switch( params.fft_placeness ) {
        case HCFFT_INPLACE:
          dtOutput = dtPlanar;
          break;

        case HCFFT_OUTOFPLACE:
          switch( params.fft_outputLayout ) {
            case HCFFT_COMPLEX_INTERLEAVED:
              dtOutput = dtComplex;
              hcKernWrite( transKernel, 0 ) << dtOutput << " *" << pmComplexOut << " = static_cast< " << dtOutput << "*> (vectArr[" << arg++ << "]);";
              break;

            case HCFFT_COMPLEX_PLANAR:
              dtOutput = dtPlanar;
              hcKernWrite( transKernel, 0 ) << dtOutput << " *" << pmRealOut << " = static_cast< " << dtOutput << "*> (vectArr[" << arg++ << "]);";
              hcKernWrite( transKernel, 0 ) << dtOutput << " *" << pmImagOut << " = static_cast< " << dtOutput << "*> (vectArr[" << arg++ << "]);";
              break;

            case HCFFT_HERMITIAN_INTERLEAVED:
            case HCFFT_HERMITIAN_PLANAR:
            case HCFFT_REAL:
            default:
              return HCFFT_INVALID;
          }

          break;

        default:
          return HCFFT_INVALID;
      }

      break;

    case HCFFT_HERMITIAN_INTERLEAVED:
    case HCFFT_HERMITIAN_PLANAR:
      return HCFFT_INVALID;

    case HCFFT_REAL:
      dtInput = dtPlanar;
      hcKernWrite( transKernel, 0 ) << dtInput << " *" << pmRealIn << " = static_cast< " << dtInput << "*> (vectArr[" << arg++ << "]);";

      switch( params.fft_placeness ) {
        case HCFFT_INPLACE:
          dtOutput = dtPlanar;
          break;

        case HCFFT_OUTOFPLACE:
          switch( params.fft_outputLayout ) {
            case HCFFT_COMPLEX_INTERLEAVED:
            case HCFFT_COMPLEX_PLANAR:
            case HCFFT_HERMITIAN_INTERLEAVED:
            case HCFFT_HERMITIAN_PLANAR:
              return HCFFT_INVALID;

            case HCFFT_REAL:
              dtOutput = dtPlanar;
              hcKernWrite( transKernel, 0 ) << dtOutput << " *" << pmRealOut << " = static_cast<" << dtOutput << "*> (vectArr[" << arg++ << "]);";
              break;

            default:
              return HCFFT_INVALID;
          }

          break;

        default:
          return HCFFT_INVALID;
      }

      break;

    default:
      return HCFFT_INVALID;
  }

  if(genTwiddle)
  {
    hcKernWrite( transKernel, 0 ) << dtComplex << " *" << TwTableLargeName() << " = static_cast<" << dtComplex << "*> (vectArr[" << arg++ << "]);";
  }

  return HCFFT_SUCCEEDS;
}


static hcfftStatus genTransposeKernel( void **twiddleslarge, accelerator acc, const hcfftPlanHandle plHandle, FFTKernelGenKeyParams & params, std::string& strKernel,
                                       const tile& lwSize, const size_t reShapeFactor, const size_t loopCount, const tile& blockSize,
                                       vector< size_t > gWorkSize, vector< size_t > lWorkSize, size_t count) {
//std::cout << " generate gcn transpose kernel "<< strKernel << std::endl;
  strKernel.reserve( 4096 );
  std::stringstream transKernel( std::stringstream::out );
  // These strings represent the various data types we read or write in the kernel, depending on how the plan
  // is configured
  std::string dtInput;        // The type read as input into kernel
  std::string dtOutput;       // The type written as output from kernel
  std::string dtPlanar;       // Fundamental type for planar arrays
  std::string dtComplex;      // Fundamental type for complex arrays
  bool genTwiddle = false;

  switch( params.fft_precision ) {
    case HCFFT_SINGLE:
      dtPlanar = "float";
      dtComplex = "float_2";
      break;

    case HCFFT_DOUBLE:
      dtPlanar = "double";
      dtComplex = "double_2";
      break;

    default:
      return HCFFT_INVALID;
      break;
  }

  //  If twiddle computation has been requested, generate the lookup function
  if(params.fft_3StepTwiddle) {
    std::string str;
    genTwiddle = true;
    if(params.fft_precision == HCFFT_SINGLE) {
    StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2, StockhamGenerator::P_SINGLE> twLarge(params.fft_N[0] * params.fft_N[1]);
    twLarge.GenerateTwiddleTable(str, plHandle);
    twLarge.TwiddleLargeAV(twiddleslarge, acc);
    } else {
      StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2, StockhamGenerator::P_DOUBLE> twLarge(params.fft_N[0] * params.fft_N[1]);
      twLarge.GenerateTwiddleTable(str, plHandle);
      twLarge.TwiddleLargeAV(twiddleslarge, acc);
    }

    hcKernWrite( transKernel, 0 ) << str << std::endl;
    hcKernWrite( transKernel, 0 ) << std::endl;
  }

  if( params.fft_placeness == HCFFT_INPLACE) {
    return HCFFT_INVALID;
  }

  for(size_t bothDir = 0; bothDir < 2; bothDir++) {
    //  Generate the kernel entry point and parameter list
    //

    bool fwd = bothDir ? false : true;
    std::string funcName;

    if(params.fft_3StepTwiddle) {
      funcName = fwd ? "transpose_gcn_tw_fwd" : "transpose_gcn_tw_back";
    } else {
      funcName = "transpose_gcn";
    }

    funcName += SztToStr(count);
    genTransposePrototype( params, lwSize, dtPlanar, dtComplex, funcName, transKernel, dtInput, dtOutput, genTwiddle);
    hcKernWrite( transKernel, 3 ) << "\thc::extent<2> grdExt( ";
    hcKernWrite( transKernel, 3 ) <<  SztToStr(gWorkSize[0]) << ", " << SztToStr(gWorkSize[1]) << "); \n" << "\thc::tiled_extent<2> t_ext = grdExt.tile(";
    hcKernWrite( transKernel, 3 ) <<  SztToStr(lwSize.x) << ", " << SztToStr(lwSize.y) << ");\n";
    hcKernWrite( transKernel, 3 ) << "\thc::parallel_for_each(acc_view, t_ext, [=] (hc::tiled_index<2> tidx) [[hc]]\n\t { ";
    hcKernWrite( transKernel, 3 ) << "const uint_2 localIndex( tidx.local[0] , tidx.local[1]); " << std::endl;
    hcKernWrite( transKernel, 3 ) << "const uint_2 localExtent( tidx.tile_dim[0], tidx.tile_dim[1]); " << std::endl;
    hcKernWrite( transKernel, 3 ) << "const uint_2 groupIndex(tidx.tile[0] , tidx.tile[1]);" << std::endl;
    hcKernWrite( transKernel, 3 ) << std::endl;
    hcKernWrite( transKernel, 3 ) << "// Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block" << std::endl;
    hcKernWrite( transKernel, 3 ) << "// Transpose of input & output blocks happens with the Offset calculation" << std::endl;
    hcKernWrite( transKernel, 3 ) << "const size_t reShapeFactor = " << reShapeFactor << ";" << std::endl;
    hcKernWrite( transKernel, 3 ) << "const size_t wgUnroll = " << loopCount << ";" << std::endl;
    hcKernWrite( transKernel, 3 ) << "const uint_2 wgTileExtent (localExtent.x * reShapeFactor, localExtent.y / reShapeFactor );" << std::endl;
    // This is the size of a matrix in the y dimension in units of group size; used to calculate stride[2] indexing
    //size_t numGroupsY = DivRoundingUp( params.fft_N[ 1 ], lwSize.y / reShapeFactor * loopCount );
    //numGroupY_1 is the number of cumulative work groups up to 1st dimension
    //numGroupY_2 is the number of cumulative work groups up to 2nd dimension and so forth
    size_t numGroupsTemp;

    if(params.transOutHorizontal) {
      numGroupsTemp = DivRoundingUp( params.fft_N[0], blockSize.x );
    } else {
      numGroupsTemp = DivRoundingUp( params.fft_N[1], blockSize.y );
    }

    hcKernWrite( transKernel, 3 ) << "const size_t numGroupsY_1" << " = " << numGroupsTemp << ";" << std::endl;

    for(int i = 2; i < params.fft_DataDim - 1; i++) {
      numGroupsTemp *= params.fft_N[i];
      hcKernWrite( transKernel, 3 ) << "const size_t numGroupsY_" << i << " = " << numGroupsTemp << ";" << std::endl;
    }

    // Generate the amount of local data share we need
    // Assumption: Even for planar data, we will still store values in LDS as interleaved
    tile ldsSize = { blockSize.x, blockSize.y};

    switch( params.fft_outputLayout ) {
      case HCFFT_COMPLEX_INTERLEAVED:
      case HCFFT_COMPLEX_PLANAR:
        hcKernWrite( transKernel, 3 ) << "// LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];" << std::endl;
        hcKernWrite( transKernel, 3 ) << "tile_static " << dtComplex << " lds[ " << ldsSize.x << " ][ " << ldsSize.y << " ];" << std::endl << std::endl;
        break;

      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;

      case HCFFT_REAL:
        hcKernWrite( transKernel, 3 ) << "tile_static " << dtPlanar << " lds[ " << ldsSize.x << " ][ " << ldsSize.y << " ];" << std::endl << std::endl;
        break;
    }

    hcKernWrite( transKernel, 3 ) << "size_t currDimIndex;" << std::endl ;
    hcKernWrite( transKernel, 3 ) << "size_t rowSizeinUnits;" << std::endl << std::endl ;
    OffsetCalc(transKernel, params, true);

    switch( params.fft_inputLayout ) {
      case HCFFT_COMPLEX_INTERLEAVED:
      	hcKernWrite( transKernel, 3 ) << "uint inOffset = iOffset;" << std::endl;
        break;

      case HCFFT_COMPLEX_PLANAR:
	      hcKernWrite( transKernel, 3 ) << "uint inOffset = iOffset;" << std::endl;
        break;

      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;

      case HCFFT_REAL:
	      hcKernWrite( transKernel, 3 ) << "uint inOffset = iOffset;" << std::endl;
        break;
    }

    // This is the loop reading through the Tile
    if( params.fft_inputLayout == HCFFT_REAL ) {
      hcKernWrite( transKernel, 3 ) << dtPlanar << " tmp;" << std::endl;
    } else {
      hcKernWrite( transKernel, 3 ) << dtComplex << " tmp;" << std::endl;
    }

    hcKernWrite( transKernel, 3 ) << "rowSizeinUnits = " << params.fft_inStride[ 1 ] << ";" << std::endl;
    hcKernWrite( transKernel, 3 ) << std::endl << std::endl;
    //
    // Group index traversal is logical where X direction is horizontal in input buffer and vertical in output buffer
    // when transOutHorizontal is enabled X direction is vertical in input buffer and horizontal in output buffer
    // Not to be confused within a tile, where X is horizontal in input and vertical in output always
    bool branchingInGroupX = params.transOutHorizontal ? ((params.fft_N[1] % blockSize.y) != 0) : ((params.fft_N[0] % blockSize.x) != 0);
    bool branchingInGroupY = params.transOutHorizontal ? ((params.fft_N[0] % blockSize.x) != 0) : ((params.fft_N[1] % blockSize.y) != 0);
    bool branchingInBoth = branchingInGroupX && branchingInGroupY;
    bool branchingInAny = branchingInGroupX || branchingInGroupY;
    size_t branchBlocks = branchingInBoth ? 4 : ( branchingInAny ? 2 : 1 );
    size_t cornerGroupX = params.transOutHorizontal ? (params.fft_N[1] / blockSize.y) : (params.fft_N[0] / blockSize.x);
    size_t cornerGroupY = params.transOutHorizontal ? (params.fft_N[0] / blockSize.x) : (params.fft_N[1] / blockSize.y);
    std::string gIndexX = "groupIndex.x"; //params.transOutHorizontal ? "currDimIndex" : "groupIndex.x";
    std::string gIndexY = "currDimIndex"; //params.transOutHorizontal ? "groupIndex.x" : "currDimIndex";
    std::string wIndexX = params.transOutHorizontal ? "yInd" : "xInd";
    std::string wIndexY = params.transOutHorizontal ? "xInd" : "yInd";
    size_t wIndexXEnd = params.transOutHorizontal ? params.fft_N[1] % blockSize.y : params.fft_N[0] % blockSize.x;
    size_t wIndexYEnd = params.transOutHorizontal ? params.fft_N[0] % blockSize.x : params.fft_N[1] % blockSize.y;

		for(size_t i = 0; i<branchBlocks; i++)
		{
			if(branchingInBoth)
      {
				if(i == 0)
				{
					hcKernWrite( transKernel, 3 ) << "if( (" << gIndexX << " == " << 
						cornerGroupX << ") && (" << gIndexY << " == " <<
						cornerGroupY << ") )" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					if(!cornerGroupY) continue;

					hcKernWrite( transKernel, 3 ) << "else if( " << gIndexX << " == " << 
						cornerGroupX << " )" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 2)
				{
					if(!cornerGroupX) continue;

					hcKernWrite( transKernel, 3 ) << "else if( " << gIndexY << " == " <<
						cornerGroupY << " )" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else
				{
					if( (!cornerGroupX) || (!cornerGroupY) ) continue;

					hcKernWrite( transKernel, 3 ) << "else" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
      }
			else if(branchingInAny)
      {
				if(i == 0)
				{
					if(branchingInGroupX)
					{
						hcKernWrite( transKernel, 3 ) << "if( " << gIndexX << " == " << 
							cornerGroupX << " )" << std::endl;
						hcKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
					else
					{
						hcKernWrite( transKernel, 3 ) << "if( " << gIndexY << " == " <<
							cornerGroupY << " )" << std::endl;
						hcKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
				}
				else
				{
					if( (!cornerGroupX) || (!cornerGroupY) ) continue;

					hcKernWrite( transKernel, 3 ) << "else" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
      }

			hcKernWrite( transKernel, 6 ) << "for( uint t=0; t < wgUnroll; t++ )" << std::endl;
			hcKernWrite( transKernel, 6 ) << "{" << std::endl;

			hcKernWrite( transKernel, 9 ) << "size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); " << std::endl;
			hcKernWrite( transKernel, 9 ) << "size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; " << std::endl;

			// Calculating the index seperately enables easier debugging through tools
			hcKernWrite( transKernel, 9 ) << "size_t gInd = xInd + rowSizeinUnits * yInd;" << std::endl;


			if(branchingInBoth)
			{
				if(i == 0)
				{
					hcKernWrite( transKernel, 9 ) << std::endl;
					hcKernWrite( transKernel, 9 ) << "if( (" << wIndexX << "< " << wIndexXEnd << ") && (" << wIndexY << " < " << wIndexYEnd << ") )" << std::endl;
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					hcKernWrite( transKernel, 9 ) << std::endl;
					hcKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexXEnd << ") )" << std::endl;
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else if(i == 2)
				{
					hcKernWrite( transKernel, 9 ) << std::endl;
					hcKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexYEnd << ") )" << std::endl;
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
			}
			else if(branchingInAny)
			{
				if(i == 0)
				{
					if(branchingInGroupX)
					{
						hcKernWrite( transKernel, 9 ) << std::endl;
						hcKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexXEnd << ") )" << std::endl;
						hcKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
					else
					{
						hcKernWrite( transKernel, 9 ) << std::endl;
						hcKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexYEnd << ") )" << std::endl;
						hcKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
				}
				else
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
			}

			switch( params.fft_inputLayout )
			{
			case HCFFT_COMPLEX_INTERLEAVED:
				{
				  hcKernWrite( transKernel, 9 ) << "tmp = pmComplexIn[ gInd + inOffset];" << std::endl;
				}
				break;
			case HCFFT_COMPLEX_PLANAR:
				{
					hcKernWrite( transKernel, 9 ) << "tmp.x = pmRealIn[ gInd + inOffset];" << std::endl;
					hcKernWrite( transKernel, 9 ) << "tmp.y = pmImagIn[ gInd + inOffset];" << std::endl;
				}
				break;
			case HCFFT_HERMITIAN_INTERLEAVED:
			case HCFFT_HERMITIAN_PLANAR:
				return HCFFT_INVALID;
			case HCFFT_REAL:
				hcKernWrite( transKernel, 9 ) << "tmp = pmRealIn[ gInd + inOffset];" << std::endl;
				break;
			}

			hcKernWrite( transKernel, 9 ) << "// Transpose of Tile data happens here" << std::endl;

			// If requested, generate the Twiddle math to multiply constant values
			if( params.fft_3StepTwiddle )
				genTwiddleMath(plHandle, params, transKernel, dtComplex, fwd );

			hcKernWrite( transKernel, 9 ) << "lds[ xInd ][ yInd ] = tmp; " << std::endl;

			if(branchingInAny)
			{
				hcKernWrite( transKernel, 9 ) << "}" << std::endl;
				hcKernWrite( transKernel, 9 ) << std::endl;
			}

			hcKernWrite( transKernel, 6 ) << "}" << std::endl;

			if(branchingInAny)
				hcKernWrite( transKernel, 3 ) << "}" << std::endl;
		}

    hcKernWrite( transKernel, 3 ) << std::endl;
    hcKernWrite( transKernel, 3 ) << "tidx.barrier.wait();" << std::endl;
    hcKernWrite( transKernel, 3 ) << std::endl;
    OffsetCalc(transKernel, params, false);

    switch( params.fft_outputLayout ) {
      case HCFFT_COMPLEX_INTERLEAVED:
     	  hcKernWrite( transKernel, 3 ) << "uint outOffset = oOffset;" << std::endl << std::endl;
        break;

      case HCFFT_COMPLEX_PLANAR:
       	hcKernWrite( transKernel, 3 ) << "uint outOffset = oOffset;" << std::endl;
        break;

      case HCFFT_HERMITIAN_INTERLEAVED:
      case HCFFT_HERMITIAN_PLANAR:
        return HCFFT_INVALID;

      case HCFFT_REAL:
	      hcKernWrite( transKernel, 3 ) << "uint outOffset = oOffset;" << std::endl << std::endl;
        break;
    }

    // Write the transposed values from LDS into global memory
    hcKernWrite( transKernel, 3 ) << "rowSizeinUnits = " << params.fft_outStride[ 1 ] << ";" << std::endl;
    hcKernWrite( transKernel, 3 ) << "const size_t transposeRatio = wgTileExtent.x / ( wgTileExtent.y * wgUnroll );" << std::endl;
    hcKernWrite( transKernel, 3 ) << "const size_t groupingPerY = wgUnroll / wgTileExtent.y;" << std::endl;
    hcKernWrite( transKernel, 3 ) << std::endl << std::endl;

		for(size_t i = 0; i<branchBlocks; i++)
		{
			if(branchingInBoth)
      {
				if(i == 0)
				{
					hcKernWrite( transKernel, 3 ) << "if( (" << gIndexX << " == " << 
						cornerGroupX << ") && (" << gIndexY << " == " <<
						cornerGroupY << ") )" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					if(!cornerGroupY) continue;

					hcKernWrite( transKernel, 3 ) << "else if( " << gIndexX << " == " << 
						cornerGroupX << " )" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else if(i == 2)
				{
					if(!cornerGroupX) continue;

					hcKernWrite( transKernel, 3 ) << "else if( " << gIndexY << " == " <<
						cornerGroupY << " )" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
				else
				{
					if( (!cornerGroupX) || (!cornerGroupY) ) continue;

					hcKernWrite( transKernel, 3 ) << "else" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
      }
			else if(branchingInAny)
      {
				if(i == 0)
				{
					if(branchingInGroupX)
					{
						hcKernWrite( transKernel, 3 ) << "if( " << gIndexX << " == " << 
							cornerGroupX << " )" << std::endl;
						hcKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
					else
					{
						hcKernWrite( transKernel, 3 ) << "if( " << gIndexY << " == " <<
							cornerGroupY << " )" << std::endl;
						hcKernWrite( transKernel, 3 ) << "{" << std::endl;
					}
				}
				else
				{
					if( (!cornerGroupX) || (!cornerGroupY) ) continue;

					hcKernWrite( transKernel, 3 ) << "else" << std::endl;
					hcKernWrite( transKernel, 3 ) << "{" << std::endl;
				}
      }

			hcKernWrite( transKernel, 6 ) << "for( uint t=0; t < wgUnroll; t++ )" << std::endl;
			hcKernWrite( transKernel, 6 ) << "{" << std::endl;
			hcKernWrite( transKernel, 9 ) << "size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % groupingPerY ); " << std::endl;
			hcKernWrite( transKernel, 9 ) << "size_t yInd = localIndex.y/groupingPerY + t * (wgTileExtent.y * transposeRatio); " << std::endl;
			hcKernWrite( transKernel, 9 ) << "tmp = lds[ yInd ][ xInd ]; " << std::endl;
			hcKernWrite( transKernel, 9 ) << "size_t gInd = xInd + rowSizeinUnits * yInd;" << std::endl;

			if(branchingInBoth)
			{
				if(i == 0)
				{
					hcKernWrite( transKernel, 9 ) << std::endl;
					hcKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexXEnd << ") && (" << wIndexX << " < " << wIndexYEnd << ") )" << std::endl;
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else if(i == 1)
				{
					hcKernWrite( transKernel, 9 ) << std::endl;
					hcKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexXEnd << ") )" << std::endl;
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;

				}
				else if(i == 2)
				{
					hcKernWrite( transKernel, 9 ) << std::endl;
					hcKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexYEnd << ") )" << std::endl;
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
				}
				else
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
			}
			else if(branchingInAny)
			{
				std::string limitToWGForRealSpecial = params.transOutHorizontal ? "groupIndex.x" : "currDimIndex";

				if(i == 0)
				{
					if(branchingInGroupX)
					{
						hcKernWrite( transKernel, 9 ) << std::endl;
						if(params.fft_realSpecial)
						{
							hcKernWrite( transKernel, 9 ) << "if( ((" << wIndexY << " == " << wIndexXEnd - 1 << ") && (" <<
								wIndexX << " < 1) && (" << limitToWGForRealSpecial << " == 0)) ";
							if(wIndexXEnd > 1)
							{
								hcKernWrite( transKernel, 0 ) << "|| (" << wIndexY << " < " << wIndexXEnd - 1 << ") )" << std::endl;
							}
							else
							{
								hcKernWrite( transKernel, 0 ) << ")" << std::endl;
							}
						}
						else
						{
							hcKernWrite( transKernel, 9 ) << "if( (" << wIndexY << " < " << wIndexXEnd << ") )" << std::endl;
						}
						hcKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
					else
					{
						hcKernWrite( transKernel, 9 ) << std::endl;
						if(params.fft_realSpecial)
						{
							hcKernWrite( transKernel, 9 ) << "if( ((" << wIndexX << " == " << wIndexYEnd - 1 << ") && (" <<
								wIndexY << " < 1) && (" << limitToWGForRealSpecial << " == 0)) ";
							if(wIndexYEnd > 1)
							{
								hcKernWrite( transKernel, 0 ) << "|| (" << wIndexX << " < " << wIndexYEnd - 1 << ") )" << std::endl;
							}
							else
							{
								hcKernWrite( transKernel, 0 ) << ")" << std::endl;
							}
						}
						else
						{
							hcKernWrite( transKernel, 9 ) << "if( (" << wIndexX << " < " << wIndexYEnd << ") )" << std::endl;
						}
						hcKernWrite( transKernel, 9 ) << "{" << std::endl;
					}
				}
				else
					hcKernWrite( transKernel, 9 ) << "{" << std::endl;
			}

			switch( params.fft_outputLayout )
			{
			case HCFFT_COMPLEX_INTERLEAVED:
				hcKernWrite( transKernel, 9 ) << "pmComplexOut[ gInd + outOffset] = tmp;" << std::endl;
				break;
			case HCFFT_COMPLEX_PLANAR:
				hcKernWrite( transKernel, 9 ) << "pmRealOut[ gInd + outOffset] = tmp.x;" << std::endl;
				hcKernWrite( transKernel, 9 ) << "pmImagOut[ gInd + outOffset] = tmp.y;" << std::endl;
				break;
			case HCFFT_HERMITIAN_INTERLEAVED:
			case HCFFT_HERMITIAN_PLANAR:
				return HCFFT_INVALID;
			case HCFFT_REAL:
				hcKernWrite( transKernel, 9 ) << "pmRealOut[ gInd + outOffset] = tmp;" << std::endl;
				break;
			}

			if(branchingInAny)
			{
				hcKernWrite( transKernel, 9 ) << "}" << std::endl;
			}

			hcKernWrite( transKernel, 6 ) << "}" << std::endl;

			if(branchingInAny)
				hcKernWrite( transKernel, 3 ) << "}" << std::endl;
		}

    hcKernWrite( transKernel, 0 ) << "}).wait();\n}}\n" << std::endl;

		strKernel = transKernel.str( );

		if(!params.fft_3StepTwiddle)
			break;
	}

  return HCFFT_SUCCEEDS;
}

template<>
hcfftStatus FFTPlan::GetKernelGenKeyPvt<Transpose_GCN> (FFTKernelGenKeyParams & params) const {
  params.fft_precision    = this->precision;
  params.fft_placeness    = this->location;
  params.fft_inputLayout  = this->ipLayout;
  params.fft_outputLayout = this->opLayout;
  params.fft_3StepTwiddle = false;
  params.fft_realSpecial  = this->realSpecial;
  params.transOutHorizontal = this->transOutHorizontal;  // using the twiddle front flag to specify horizontal write
  // we do this so as to reuse flags in FFTKernelGenKeyParams
  // and to avoid making a new one
  ARG_CHECK( this->inStride.size( ) == this->outStride.size( ) );

  if( HCFFT_INPLACE == params.fft_placeness ) {
    //  If this is an in-place transform the
    //  input and output layout, dimensions and strides
    //  *MUST* be the same.
    //
    ARG_CHECK( params.fft_inputLayout == params.fft_outputLayout )

    for( size_t u = this->inStride.size(); u-- > 0; ) {
      ARG_CHECK( this->inStride[u] == this->outStride[u] );
    }
  }

  params.fft_DataDim = this->length.size() + 1;
  int i = 0;

  for(i = 0; i < (params.fft_DataDim - 1); i++) {
    params.fft_N[i]         = this->length[i];
    params.fft_inStride[i]  = this->inStride[i];
    params.fft_outStride[i] = this->outStride[i];
  }

  params.fft_inStride[i]  = this->iDist;
  params.fft_outStride[i] = this->oDist;

  if (this->large1D != 0) {
    ARG_CHECK (params.fft_N[0] != 0)
    ARG_CHECK ((this->large1D % params.fft_N[0]) == 0)
    params.fft_3StepTwiddle = true;
    ARG_CHECK ( this->large1D  == (params.fft_N[1] * params.fft_N[0]) );
  }

  //  Query the devices in this context for their local memory sizes
  //  How we generate a kernel depends on the *minimum* LDS size for all devices.
  //
  const FFTEnvelope* pEnvelope = NULL;
  this->GetEnvelope( &pEnvelope );
  BUG_CHECK( NULL != pEnvelope );
  // TODO:  Since I am going with a 2D workgroup size now, I need a better check than this 1D use
  // Check:  CL_DEVICE_MAX_WORK_GROUP_SIZE/CL_KERNEL_WORK_GROUP_SIZE
  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  params.fft_R = 1; // Dont think i'll use
  params.fft_SIMD = pEnvelope->limit_WorkGroupSize; // Use devices maximum workgroup size
	params.limit_LocalMemSize = this->envelope.limit_LocalMemSize;

  return HCFFT_SUCCEEDS;
}

// Constants that specify the bounding sizes of the block that each workgroup will transpose
static const tile lwSize = { {16}, {16} };
static const size_t reShapeFactor = 4;   // wgTileSize = { lwSize.x * reShapeFactor, lwSize.y / reShapeFactor }

static hcfftStatus CalculateBlockSize(const hcfftPrecision precision, size_t &loopCount, tile &blockSize)
{
    switch( precision )
    {
    case HCFFT_SINGLE:
        loopCount = 16;
        break;
    case HCFFT_DOUBLE:
        // Double precisions need about half the amount of LDS space as singles do
        loopCount = 8;
        break;
    default:
        return HCFFT_INVALID;
        break;
    }

	blockSize.x = lwSize.x * reShapeFactor;
	blockSize.y = lwSize.y / reShapeFactor * loopCount;

	return HCFFT_SUCCEEDS;
}

template<>
hcfftStatus FFTPlan::GetWorkSizesPvt<Transpose_GCN> (std::vector<size_t> & globalWS, std::vector<size_t> & localWS) const {
  FFTKernelGenKeyParams fftParams;
  this->GetKernelGenKeyPvt<Transpose_GCN>( fftParams );

	size_t loopCount = 0;
	tile blockSize = {0, 0};
	CalculateBlockSize(fftParams.fft_precision, loopCount, blockSize);

  // We need to make sure that the global work size is evenly divisible by the local work size
  // Our transpose works in tiles, so divide tiles in each dimension to get count of blocks, rounding up for remainder items
  size_t numBlocksX = fftParams.transOutHorizontal ?
                      DivRoundingUp(fftParams.fft_N[ 1 ], blockSize.y ) :
                      DivRoundingUp(fftParams.fft_N[ 0 ], blockSize.x );
  size_t numBlocksY = fftParams.transOutHorizontal ?
                      DivRoundingUp( fftParams.fft_N[ 0 ], blockSize.x ) :
                      DivRoundingUp( fftParams.fft_N[ 1 ], blockSize.y );
  size_t numWIX = numBlocksX * lwSize.x;
  // Batches of matrices are lined up along the Y axis, 1 after the other
  size_t numWIY = numBlocksY * lwSize.y * this->batchSize;

  // fft_DataDim has one more dimension than the actual fft data, which is devoted to batch.
  // dim from 2 to fft_DataDim - 2 are lined up along the Y axis
  for(int i = 2; i < fftParams.fft_DataDim - 1; i++) {
    numWIY *= fftParams.fft_N[i];
  }

  globalWS.clear( );
  globalWS.push_back( numWIX );
  globalWS.push_back( numWIY );
  localWS.clear( );
  localWS.push_back( lwSize.x );
  localWS.push_back( lwSize.y );
  return HCFFT_SUCCEEDS;
}

//  OpenCL does not take unicode strings as input, so this routine returns only ASCII strings
//  Feed this generator the FFTPlan, and it returns the generated program as a string
template<>
hcfftStatus FFTPlan::GenerateKernelPvt<Transpose_GCN>(const hcfftPlanHandle plHandle, FFTRepo& fftRepo, size_t count, bool exist) const {
  FFTKernelGenKeyParams fftParams;
  this->GetKernelGenKeyPvt<Transpose_GCN>( fftParams );

  if(!exist)
  {
  	size_t loopCount = 0;
	  tile blockSize = {0, 0};
  	CalculateBlockSize(fftParams.fft_precision, loopCount, blockSize);

    vector< size_t > gWorkSize;
    vector< size_t > lWorkSize;
    this->GetWorkSizesPvt<Transpose_GCN> (gWorkSize, lWorkSize);
    std::string programHeader, programCode;
    programHeader = hcHeader();
    genTransposeKernel((void**)&twiddleslarge, acc, plHandle, fftParams, programCode, lwSize, reShapeFactor, loopCount, blockSize, gWorkSize, lWorkSize, count);
    programHeader += programCode;
    fftRepo.setProgramCode(Transpose_GCN, plHandle, fftParams, programHeader);

    // Note:  See genFunctionPrototype( )
    if( fftParams.fft_3StepTwiddle ) {
      fftRepo.setProgramEntryPoints( Transpose_GCN, plHandle, fftParams, "transpose_gcn_tw_fwd", "transpose_gcn_tw_back");
    } else {
      fftRepo.setProgramEntryPoints( Transpose_GCN, plHandle, fftParams, "transpose_gcn", "transpose_gcn");
    }
  }
  else
  {
    size_t large1D = 0;

    if(fftParams.fft_realSpecial) {
      large1D = fftParams.fft_N[0] * fftParams.fft_realSpecial_Nr;
    } else {
      large1D = fftParams.fft_N[0] * fftParams.fft_N[1];
    }

    if(fftParams.fft_precision == HCFFT_SINGLE)
    {
      // twiddle factors for 1d-large 3-step algorithm
      if(fftParams.fft_3StepTwiddle && !twiddleslarge) {
        StockhamGenerator::TwiddleTableLarge<hc::short_vector::float_2, StockhamGenerator::P_SINGLE> twLarge(large1D);
        twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
      }
    }
    else
    {
      // twiddle factors for 1d-large 3-step algorithm
      if(fftParams.fft_3StepTwiddle && !twiddleslarge) {
        StockhamGenerator::TwiddleTableLarge<hc::short_vector::double_2, StockhamGenerator::P_DOUBLE> twLarge(large1D);
        twLarge.TwiddleLargeAV((void**)&twiddleslarge, acc);
      }
    }
  }
  return HCFFT_SUCCEEDS;
}
