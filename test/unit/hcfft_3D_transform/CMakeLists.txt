# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

file(GLOB SRCS *.cpp)

# Choice to take compilation flags from source or package
if(EXISTS ${MCWHCCBUILD})
  execute_process(COMMAND ${CLAMP_CONFIG} --build --cxxflags
                         OUTPUT_VARIABLE CLAMP_CXXFLAGS)
  execute_process(COMMAND ${CLAMP_CONFIG} --build --ldflags --gtest 
                            OUTPUT_VARIABLE CLAMP_LDFLAGS)
else(EXISTS ${MCWHCCBUILD})
  execute_process(COMMAND ${CLAMP_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE CLAMP_CXXFLAGS)
  execute_process(COMMAND ${CLAMP_CONFIG}  --install --ldflags --gtest
                            OUTPUT_VARIABLE CLAMP_LDFLAGS)
endif(EXISTS ${MCWHCCBUILD})


string(STRIP "${CLAMP_CXXFLAGS}" CLAMP_CXXFLAGS)
string(STRIP "${CLAMP_LDFLAGS}" CLAMP_LDFLAGS)
set (CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS}")
set (CLAMP_LDFLAGS "${CLAMP_LDFLAGS}")

SET_PROPERTY(SOURCE ${SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS} -DGTEST_HAS_TR1_TUPLE=0 -I$ENV{OPENCL_INCLUDE_PATH} ")
ADD_EXECUTABLE(test_3D_transform ${SRCS})
SET_PROPERTY(TARGET test_3D_transform APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} -L$ENV{CLFFT_LIBRARY_PATH} -lclFFT -L$ENV{OPENCL_LIBRARY_PATH} -lOpenCL")

TARGET_LINK_LIBRARIES(test_3D_transform hcfft)

add_test(NAME test3DTransform COMMAND test_3D_transform)

