#Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET (TESTSRCS
    2D_C2C.cpp 2D_C2R.cpp 2D_D2Z.cpp
    2D_R2C.cpp 2D_Z2D.cpp 2D_Z2Z.cpp
)

set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
	set (HIP_PATH /opt/rocm/hip)
endif()

set (HIPFFT_ROOT $ENV{HIPFFT_ROOT})
if (NOT DEFINED HIPFFT_ROOT)
	set (HIPFFT_ROOT ../../../)
endif()
include_directories(${HIPFFT_ROOT}/lib/include)

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)

MESSAGE ("HIP_PATH=" ${HIP_PATH})

if (${HIP_PLATFORM} STREQUAL "hcc")
	MESSAGE ("HIP_PLATFORM=hcc")
	set (HSA_PATH /opt/hsa)

	# Add HSA library:
#	add_library(hsa-runtime64 SHARED IMPORTED)
#	set_property(TARGET hsa-runtime64 PROPERTY IMPORTED_LOCATION "${HSA_PATH}/lib/libhsa-runtime64.so")

	set (HCFFT_ROOT $ENV{HCFFT_ROOT})
	if (NOT DEFINED HCFFT_ROOT)
		set (HCFFT_ROOT ../../)
	endif()

	#These includes are used for all files.
	#Include HIP and HC since the tests need both of these:
	#Note below HSA path is surgically included only where necessary.
	include_directories(${HIP_PATH}/include ${HC_PATH}/lib/include  ${HCFFT_ROOT}/lib/include  )

	# hip_hcc.o:
#	add_library(hip_hcc OBJECT ${HIP_PATH}/src/hip_hcc.cpp)
#	target_include_directories(hip_hcc PRIVATE ${HSA_PATH}/include)

        add_library(hcfft SHARED IMPORTED)
        set_target_properties( hcfft PROPERTIES IMPORTED_LOCATION "${HCFFT_ROOT}/libhcfft.so" )

        MESSAGE ("CURRENTSRC${CMAKE_CURRENT_SOURCE_DIR}")
	SET(HCFFT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/include/")
        SET(HCFFT_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/src")

  	set (HCC_CXXFLAGS "-I${HCFFT_INCLUDE_PATH} -I${CMAKE_CURRENT_SOURCE_DIR}/../../lib/include/hcc_detail/")
 	set (HCC_LDFLAGS "-L${HCFFT_LIBRARY_PATH}")
  	set (LINK "-lhcfft")

elseif (${HIP_PLATFORM} STREQUAL "nvcc")

	MESSAGE ("HIP_PLATFORM = nvcc")
	#NVCC does not support -rdynamic option 
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS )
	set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS )

	find_package( CUDA REQUIRED )

else()
	MESSAGE (FATAL_ERROR "UNKNOWN HIP_PLATFORM=" ${HIP_PLATFORM})
endif()

set (HIPCC ${HIP_PATH}/bin/hipcc)
set (CMAKE_CXX_COMPILER ${HIPCC})
add_definitions(-stdlib=libc++) 

macro (make_hip_executable exe cpp)  
  if (${HIP_PLATFORM} STREQUAL "hcc")
    add_executable (bin/${exe} ${cpp} )
    TARGET_LINK_LIBRARIES(bin/${exe} ${LINK} ${HCC_LDFLAGS})
  else()
    EXECUTE_PROCESS (COMMAND nvcc ${cpp} -lcublas -I${HIPBLAS_ROOT}/include -I${HIP_PATH}/include -o bin/${exe})
  endif() 
endmacro()

if (${HIP_PLATFORM} STREQUAL "hcc")
	FOREACH(test_file ${TESTSRCS})
	  get_filename_component (name_without_extension ${test_file} NAME_WE)
	  make_hip_executable(${name_without_extension} ${test_file})
	ENDFOREACH()	
endif()



