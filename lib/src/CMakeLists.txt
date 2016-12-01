CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

# Find HIP
set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
	set (HIP_PATH /opt/rocm/hip)
endif()

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)
MESSAGE ("HIP_PATH=" ${HIP_PATH})

#Setting a Project name
SET(PROJECT_NAME "hcfft")
SET(PROJECT_NAME_EXT "hipfft")
PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")

if (${HIP_PLATFORM} STREQUAL "hcc")
#Setting a variable for source files
FILE(GLOB HCFFTSRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
LIST(REMOVE_ITEM HCFFTSRCS ${CMAKE_CURRENT_SOURCE_DIR}/hipfft.cpp)
SET(HIPFFTSRCS ${HCFFTSRCS} ${CMAKE_CURRENT_SOURCE_DIR}/hipfft.cpp)

execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
execute_process(COMMAND ${HCC_CONFIG} --install --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)

string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${HIP_PATH}/include -D__HIP_PLATFORM_HCC__=")
set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HIP_PATH}/lib -lhip_hcc")


# Setting header dependencies for hcfft library generation
FOREACH(src_file ${HCFFTSRCS})
  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
ENDFOREACH()

# Setting header dependencies for hipfft library generation
FOREACH(src_file ${HIPFFTSRCS})
  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
ENDFOREACH()

#Generating hcfft shared object
ADD_LIBRARY("${PROJECT_NAME}" SHARED  ${HCFFTSRCS})
SET_PROPERTY(TARGET "${PROJECT_NAME}" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
TARGET_LINK_LIBRARIES("${PROJECT_NAME}" hc_am)

#Generating hipfft shared object
ADD_LIBRARY("${PROJECT_NAME_EXT}" SHARED ${HIPFFTSRCS})
SET_PROPERTY(TARGET "${PROJECT_NAME_EXT}" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
TARGET_LINK_LIBRARIES("${PROJECT_NAME_EXT}" hc_am)

INSTALL(TARGETS "${PROJECT_NAME}" 
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

INSTALL(TARGETS "${PROJECT_NAME_EXT}" 
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB hcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/hcc_detail/*.h")
file(GLOB nvcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvcc_detail/*.h")
INSTALL(FILES ${headers} DESTINATION include)
INSTALL(FILES ${hcc_detail} DESTINATION include/hcc_detail)
INSTALL(FILES ${nvcc_detail} DESTINATION include/nvcc_detail)

elseif (${HIP_PLATFORM} STREQUAL "nvcc")
#/opt/rocm/hip/bin/hipcc -Xcompiler -fpic -c -I/home/santhosh/hcfft/lib/include/ -I/home/santhosh/hcfft/lib/include/nvcc_detail/  hipfft.cpp -o hip_obj.o -lcufft
#/opt/rocm/hip/bin/hipcc -shared -o libhipfft_nvcc.so hip_obj.o -lcufft 
execute_process(COMMAND ${HIP_PATH}/bin/hipcc -Xcompiler -fpic -c -I/../include/ -I/../include/nvcc_detail/ hipfft.cpp -o ../../build/lib/src/hip_obj.o -lcufft)
execute_process(COMMAND ${HIP_PATH}/bin/hipcc -shared -o ../../build/lib/src/libhipfft_nvcc.so ../../build/lib/src/hip_obj.o -lcufft )
else()
	MESSAGE (FATAL_ERROR "UNKNOWN HIP_PLATFORM=" ${HIP_PLATFORM})
endif()
