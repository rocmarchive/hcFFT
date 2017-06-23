# This script is invoked to install the hcfft library and test sources
# Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

# getconf _NPROCESSORS_ONLN
working_threads=8

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

if ( [ ! -z $HIP_PATH ] || [ -x "/opt/rocm/hip/bin/hipcc" ] ); then 
  export HIP_SUPPORT=on
elif ( [ "$platform" = "nvcc" ]); then
  echo "HIP not found. Install latest HIP to continue."
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
copt="-O3"
verbose=""
install=0

# Help menu
print_help() {
cat <<-HELP
===================================================================================================================
This script is invoked to install hcFFT library and test sources. Please provide the following arguments:

  1) ${green}--test${reset}    Test to enable the library testing.
  2) ${green}--bench${reset}   Profile benchmark using chrono timer.
  3) ${green}--hip_so${reset}  To create libhipfft.so 
  4) ${green}--debug${reset}    Compile with debug info (-g)
  5) ${green}--verbose${reset}  Run make with VERBOSE=1
===================================================================================================================
Usage: ./install.sh --test=on
===================================================================================================================
Example:
   ${green}./build.sh --test=on${reset} 
   ${green}./build.sh --bench=on${reset}
   ${green}./build.sh --hip_so=on${reset} 

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
    --debug)
      copt="-g"
      ;;
    --verbose)
      verbose="VERBOSE=1"
      ;;
    --bench=*)
      bench="${1#*=}"
      ;;
    --install)
      install="1"
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
mkdir $current_work_dir/build/packaging
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
  cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="$copt -fPIC" -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcfft $current_work_dir
  make -j$working_threads package $verbose
  make -j$working_threads $verbose

  if [ "$install" = "1" ]; then
    sudo make -j$working_threads install
  fi
  cd $build_dir/packaging/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="$copt -fPIC" -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcfft $current_work_dir/packaging/

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
   if [ $HIP_SUPPORT = "on" ]; then
     mkdir -p $current_work_dir/build/test/unit-hip/hipfft_transforms/bin/
     mkdir -p $current_work_dir/build/test/unit-hip/hipfft_Create_Destroy_Plan/bin/
   fi
   mkdir -p $current_work_dir/build/test/FFT_benchmark_Convolution_Networks/Comparison_tests/bin/
   set -e
   
   # Build Tests
   cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="$copt -fPIC" $current_work_dir/test/
   make -j$working_threads $verbose

   chmod +x $current_work_dir/test/unit-api/test.sh
   cd $current_work_dir/test/unit-api/
   # Invoke hc unit test script
   printf "* UNIT API TESTS *\n"
   printf "******************\n"
   ./test.sh
   
   if [ $HIP_SUPPORT = "on" ]; then
     chmod +x $current_work_dir/test/unit-hip/test.sh
     cd $current_work_dir/test/unit-hip/
     # Invoke hip unit test script
     printf "* UNIT HIP TESTS *\n"
     printf "******************\n"
     ./test.sh
   fi 
  fi

  if [ "$bench" = "on" ]; then #bench=on run chrono timer
    cd $current_work_dir/test/FFT_benchmark_Convolution_Networks/
    ./runme_chronotimer.sh
  fi
fi

if [ "$platform" = "nvcc" ]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$current_work_dir/build/lib/src
  cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="$copt -fPIC" -DCMAKE_INSTALL_PREFIX=/opt/rocm/hipfft $current_work_dir
  make -j$working_threads package $verbose
  make -j$working_threads $verbose
  
  cd $build_dir/packaging/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="$copt -fPIC" -DCMAKE_INSTALL_PREFIX=/opt/rocm/hipfft $current_work_dir/packaging/
  
  echo "${green}HIPFFT Build Completed!${reset}"

  if ( [ "$testing" = "on" ] ); then
    set +e
    mkdir -p $current_work_dir/build/test/unit-hip/hipfft_transforms/bin/
    mkdir -p $current_work_dir/build/test/unit-hip/hipfft_Create_Destroy_Plan/bin/
    set -e 
    
    # Build Tests
    cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="$copt -fPIC" $current_work_dir/test/
  make -j$working_threads  $verbose
    
    chmod +x $current_work_dir/test/unit-hip/test.sh
    cd $current_work_dir/test/unit-hip/
    # Invoke hip unit test script
    printf "* UNIT HIP TESTS *\n"
    printf "******************\n"
    ./test.sh
  fi
fi

# Simple test to confirm installation
#$build_dir/test/src/fft 2 12

# TODO: ADD More options to perform benchmark and testing
