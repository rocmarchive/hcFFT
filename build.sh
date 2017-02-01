# This script is invoked to install the hcfft library and test sources
# Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

# CHECK FOR COMPILER PATH
if [ ! -z $HCC_HOME ]
then
  platform="hcc"
  if [ -x "$HCC_HOME/bin/clang++" ]
  then
    cmake_c_compiler=$HCC_HOME/bin/clang
    cmake_cxx_compiler=$HCC_HOME/bin/clang++
  fi
elif [ -x "/opt/rocm/hcc/bin/clang++" ]
then
  platform="hcc"
  cmake_c_compiler=/opt/rocm/hcc/bin/clang
  cmake_cxx_compiler=/opt/rocm/hcc/bin/clang++
elif [ -x "/usr/local/cuda/bin/nvcc" ];
then
  platform="nvcc"
  cmake_c_compiler="/usr/bin/gcc"
  cmake_cxx_compiler="/usr/bin/g++"
else
  echo "Neither clang  or NVCC compiler found"
  echo "Not an AMD or NVCC compatible stack"
  exit 1
fi

if [ ! -z $HIP_PATH ]
then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HIP_PATH/lib
else
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib
fi

export CLAMP_NOTILECHECK=ON

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`
install=0

# Help menu
print_help() {
cat <<-HELP
===================================================================================================================
This script is invoked to install hcFFT library and test sources. Please provide the following arguments:

  1) ${green}--test${reset}    Test to enable the library testing.
  2) ${green}--bench${reset}   Profile benchmark using chrono timer.
  3) ${green}--hip_so${reset}  To create libhipfft.so 
===================================================================================================================
Usage: ./install.sh --test=on
===================================================================================================================
Example:
(1) ${green}./build.sh --test=on${reset} 
(2) ${green}./build.sh --bench=on${reset}
(3) ${green}./build.sh --hip_so=on${reset} 

NOTE:
${green} Install FFTW ${reset}
<sudo apt-get install libfftw3-dev>
${green} Update AMDAPPSDKROOT ${reset}
 <export AMDAPPSDKROOT=/home/user/AMDAPPSDKROOT>
===================================================================================================================
HELP
exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --test=*)
      testing="${1#*=}"
      ;;
    --bench=*)
      bench="${1#*=}"
      ;;
    --install)
      install="1"
      ;;
    --hip_so=*)
      hip_so="${1#*=}"
      ;;
    --help) print_help;;
    *)
      printf "************************************************************\n"
      printf "* Error: Invalid arguments, run --help for valid arguments.*\n"
      printf "************************************************************\n"
      exit 1
  esac
  shift
done

if ( [ "$bench" = "on" ] ); then
    export BENCH_MARK=on
fi

if [ -z $bench ]; then
    bench="off"
fi

if [ "$install" = "1" ]; then
    export INSTALL_OPT=on
fi

set +e
# MAKE BUILD DIR
mkdir -p $current_work_dir/build
mkdir -p $current_work_dir/build/lib
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build
#change to library build
cd $build_dir

if [ "$platform" = "hcc" ]; then

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$hcfft_install/lib/hcfft
  export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$hcfft_install/include/hcfft
  export HCFFT_LIBRARY_PATH=$current_work_dir/build/lib/src
  export LD_LIBRARY_PATH=$HCFFT_LIBRARY_PATH:$LD_LIBRARY_PATH

  # Cmake and make libhcfft: Install hcFFT
  if [ "$hip_so" = "on" ] ; then  
    cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DHIP_SHARED_OBJ=ON -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcfft $current_work_dir
  else
    cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcfft $current_work_dir
  fi
  make package
  make

  if [ "$install" = "1" ]; then
    sudo make install
    if  [ "$hip_so" = "on" ] ; then
     cd $build_dir/packaging/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DHIP_SHARED_OBJ=ON -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcfft $current_work_dir/packaging/
    else
     cd $build_dir/packaging/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcfft $current_work_dir/packaging/
    fi
  fi

  # KERNEL CACHE DIR
  mkdir -p $HOME/kernCache

  #Test=OFF (Build library and tests)
  if ( [ -z $testing ] ) || ( [ "$testing" = "off" ] ); then
    echo "${green}HCFFT Installation Completed!${reset}"
  # Test=ON (Build and test the library)
  elif ( [ "$testing" = "on" ] ); then
   
   set +e
   mkdir -p $current_work_dir/build/test
   mkdir -p $current_work_dir/build/test/src/bin/
   mkdir -p $current_work_dir/build/test/unit-api/hcfft_transforms/bin/
   mkdir -p $current_work_dir/build/test/unit-api/hcfft_Create_Destroy_Plan/bin/
   mkdir -p $current_work_dir/build/test/unit-hip/hipfft_transforms/bin/
   mkdir -p $current_work_dir/build/test/unit-hip/hipfft_Create_Destroy_Plan/bin/
   mkdir -p $current_work_dir/build/test/FFT_benchmark_Convolution_Networks/Comparison_tests/bin/
   set -e
   
   # Build Tests
   if  [ "$hip_so" = "on" ] ; then
     cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DHIP_SHARED_OBJ=ON -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
   else
     cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
   fi
   make

   chmod +x $current_work_dir/test/unit-api/test.sh
   cd $current_work_dir/test/unit-api/
   # Invoke hc unit test script
   printf "* UNIT API TESTS*\n"
   printf "*****************\n"
   ./test.sh
   
   chmod +x $current_work_dir/test/unit-hip/test.sh
   cd $current_work_dir/test/unit-hip/
   # Invoke hip unit test script
   printf "* UNIT HIP TESTS*\n"
   printf "*****************\n"
   ./test.sh
    
  fi

  if [ "$bench" = "on" ]; then #bench=on run chrono timer
    cd $current_work_dir/test/FFT_benchmark_Convolution_Networks/
    ./runme_chronotimer.sh
  fi
fi

if [ "$platform" = "nvcc" ]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$current_work_dir/build/lib/src
  if ( [ "$hip_so" = "on" ] ); then
    cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DHIP_SHARED_OBJ=ON -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcfft $current_work_dir
    make package
    make
    echo "${green}HIPFFT Build Completed!${reset}"
  fi

  if ( [ "$testing" = "on" ] ); then
    set +e
    mkdir -p $current_work_dir/build/test/unit-hip/hipfft_transforms/bin/
    mkdir -p $current_work_dir/build/test/unit-hip/hipfft_Create_Destroy_Plan/bin/
    set -e 
    
    # Build Tests
    if ( [ "$hip_so" = "on" ] ); then
      cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DHIP_SHARED_OBJ=ON -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="-fPIC" $current_work_dir/test/
    else
      cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="-fPIC" $current_work_dir/test/
    fi
    make
    
    chmod +x $current_work_dir/test/unit-hip/test.sh
    cd $current_work_dir/test/unit-hip/
    # Invoke hip unit test script
    printf "* UNIT HIP TESTS*\n"
    printf "*****************\n"
    ./test.sh
  fi
fi

# Simple test to confirm installation
#$build_dir/test/src/fft 2 12

# TODO: ADD More options to perform benchmark and testing
