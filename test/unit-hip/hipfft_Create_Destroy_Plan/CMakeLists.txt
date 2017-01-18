# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

SET (HIPSRCS hipfft_Create_destroy_Plan.cpp)

IF(${PLATFORM} MATCHES "hcc")
  # Find HCC compiler
  FIND_PACKAGE(HC++ 1.0 REQUIRED)
  execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                          OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG}  --install --ldflags
                          OUTPUT_VARIABLE HCC_LDFLAGS)

  SET(HIPFFT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/include/")
  SET(HIPFFT_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/src")

  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${HIPFFT_INCLUDE_PATH}")
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HIPFFT_LIBRARY_PATH}")
  INCLUDE_DIRECTORIES(${HIP_PATH}/include)
  
  IF(${HIP_SHARED_OBJ} MATCHES "ON")
    SET (LINK "-lhipfft_hcc -lm")
  ELSE()
    SET (LINK "-lhcfft -lm")
  ENDIF()

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
  FOREACH(test_file ${HIPSRCS})
    IF(${HIP_SHARED_OBJ} MATCHES "ON")
      SET_PROPERTY(SOURCE ${test_file} ../gtest_main.cpp ../gtest-all.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} -D__HIP_PLATFORM_HCC__ -DHIP_SHARED_OBJ=1")
    ELSE()
      SET_PROPERTY(SOURCE ${test_file} ../gtest_main.cpp ../gtest-all.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} -D__HIP_PLATFORM_HCC__")
    ENDIF()
    get_filename_component (name_without_extension ${test_file} NAME_WE)
    ADD_EXECUTABLE(${name_without_extension} ${test_file} ../gtest-all.cpp ../gtest_main.cpp)
    SET_PROPERTY(TARGET ${name_without_extension} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ${LINK}")
  ENDFOREACH()

ELSEIF(${PLATFORM} MATCHES "nvcc")
  SET(HIPFFT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/include/")
  SET(HIPFFT_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/src")

  set (HIP_CXXFLAGS "${HIP_CXXFLAGS} -I${HIPFFT_INCLUDE_PATH} -I/usr/local/cuda/include/ -I${HIP_PATH}/include")
  IF(${HIP_SHARED_OBJ} MATCHES "ON") 
    set (HIP_LDFLAGS "${HIP_LDFLAGS} -L${HIPFFT_LIBRARY_PATH} -L/usr/local/cuda/lib64")
  ELSE()
    set (HIP_LDFLAGS "${HIP_LDFLAGS} -L/usr/local/cuda/lib64")
  ENDIF()
  INCLUDE_DIRECTORIES(${HIP_PATH}/include)
  #SET (LINK "-lpthread")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
  FOREACH(test_file ${HIPSRCS})
    IF(${HIP_SHARED_OBJ} MATCHES "ON")
      SET_PROPERTY(SOURCE ${test_file} ../gtest_main.cpp ../gtest-all.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${HIP_CXXFLAGS} -D__HIP_PLATFORM_NVCC__ -DHIP_SHARED_OBJ=1")
    ELSE()
      SET_PROPERTY(SOURCE ${test_file} ../gtest_main.cpp ../gtest-all.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${HIP_CXXFLAGS} -D__HIP_PLATFORM_NVCC__")
    ENDIF()
    get_filename_component (name_without_extension ${test_file} NAME_WE)
    ADD_EXECUTABLE(${name_without_extension} ../gtest-all.cpp ../gtest_main.cpp ${test_file})
    SET_PROPERTY(TARGET ${name_without_extension} APPEND_STRING PROPERTY LINK_FLAGS " ${HIP_LDFLAGS} ${LINK}")
    IF(${HIP_SHARED_OBJ} MATCHES "ON")
      TARGET_LINK_LIBRARIES(${name_without_extension} cudart hipfft_nvcc pthread)
    ELSE()
      TARGET_LINK_LIBRARIES(${name_without_extension} cudart cufft pthread)
    ENDIF()
  ENDFOREACH()
ENDIF()
