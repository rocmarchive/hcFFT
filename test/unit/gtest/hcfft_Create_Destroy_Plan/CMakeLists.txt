# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

SET (HCSRCS hcfft_Create_destroy_Plan.cpp)

# Choice to take compilation flags from source or package
if(EXISTS ${MCWHCCBUILD})
  execute_process(COMMAND ${HCC_CONFIG} --build --cxxflags
                         OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG} --build --ldflags
                            OUTPUT_VARIABLE HCC_LDFLAGS)
else(EXISTS ${MCWHCCBUILD})
  execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG}  --install --ldflags
                            OUTPUT_VARIABLE HCC_LDFLAGS)
endif(EXISTS ${MCWHCCBUILD})

string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
set (HCC_CXXFLAGS "${HCC_CXXFLAGS}")
string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
set (HCC_LDFLAGS "${HCC_LDFLAGS} -L $ENV{hcfft_install}/lib/hcfft")
SET (LINK " -lhcfft -lhc_am")
FOREACH(test_file ${HCSRCS})
  SET_PROPERTY(SOURCE ${test_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} -DGTEST_HAS_TR1_TUPLE=0 -I$ENV{OPENCL_INCLUDE_PATH}")
  get_filename_component (name_without_extension ${test_file} NAME_WE)
  ADD_EXECUTABLE( bin/${name_without_extension} ${test_file} gtest-all.cpp gtest_main.cpp)
  SET_PROPERTY(TARGET bin/${name_without_extension} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} -L$ENV{CLFFT_LIBRARY_PATH} -lclFFT -L$ENV{OPENCL_LIBRARY_PATH} -lOpenCL ${LINK}")
ENDFOREACH()
