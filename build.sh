# This script is invoked to install the hcfft library and test sources
# Preliminary version

# CHECK FOR COMPILER PATH
if [ ! -z $HCCLC ]
then
  if [ -x "/opt/rocm/hcc-lc/bin/clang++" ]
  then
    cmake_c_compiler=/opt/rocm/hcc-lc/bin/clang
    cmake_cxx_compiler=/opt/rocm/hcc-lc/bin/clang++
  fi

elif [ -x "/opt/rocm/hcc-hsail/bin/clang++" ]
then
  cmake_c_compiler=/opt/rocm/hcc-hsail/bin/clang
  cmake_cxx_compiler=/opt/rocm/hcc-hsail/bin/clang++

else
  echo "Clang compiler not found"
  exit 1
fi

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# Help menu
print_help() {
cat <<-HELP
===================================================================================================================
This script is invoked to install hcFFT library and test sources. Please provide the following arguments:

  1) ${green}--test${reset}    Test to enable the library testing.
  2) ${green}--bench${reset}   Profile benchmark using chrono timer.
===================================================================================================================
Usage: ./install.sh --test=on
===================================================================================================================
Example:
(1) ${green}./install.sh --test=on${reset} (sudo access needed)
(2) ${green}./install.sh --bench=on${reset}

NOTE:
${green} Please Export CLFFT_LIBRARY_PATH to point to clFFT library ${reset}
 <export CLFFT_LIBRARY_PATH=/home/user/clFFT/build/library>
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
    --help) print_help;;
    *)
      printf "************************************************************\n"
      printf "* Error: Invalid arguments, run --help for valid arguments.*\n"
      printf "************************************************************\n"
      exit 1
  esac
  shift
done

if [ -z $bench ]; then
    bench="off"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$hcfft_installlib/hcfft
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CLFFT_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$hcfft_install/include/hcfft
export HCFFT_LIBRARY_PATH=$PWD/build/lib/src
export LD_LIBRARY_PATH=$HCFFT_LIBRARY_PATH:$LD_LIBRARY_PATH

set +e
# MAKE BUILD DIR
mkdir -p $current_work_dir/build
mkdir -p $current_work_dir/build/lib
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build

#change to library build
cd $build_dir

# Cmake and make libhcfft: Install hcFFT
cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir
make package
make

# KERNEL CACHE DIR
mkdir -p $HOME/kernCache

#Test=OFF (Build library and tests)
if ( [ -z $testing ] ) || ( [ "$testing" = "off" ] ); then
  echo "${green}HCFFT Installation Completed!${reset}"
# Test=ON (Build and test the library)
elif ( [ "$testing" = "on" ] ); then
  export OPENCL_INCLUDE_PATH=$AMDAPPSDKROOT/include
  export OPENCL_LIBRARY_PATH=$AMDAPPSDKROOT/lib/x86_64/

  set +e
  mkdir -p $current_work_dir/build/test
  mkdir -p $current_work_dir/build/test/src/bin/
  mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_1D_transform/bin/
  mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_2D_transform/bin/
  mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_3D_transform/bin/
  mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_Create_Destroy_Plan/bin/
  mkdir -p $current_work_dir/build/test/FFT_benchmark_Convolution_Networks/Comparison_tests/bin/
  set -e

  # Build Tests
  cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
  make

  chmod +x $current_work_dir/test/unit/test.sh
  cd $current_work_dir/test/unit/
# Invoke test script
  ./test.sh
fi

if [ "$bench" = "on" ]; then #bench=on run chrono timer
  cd $current_work_dir/test/FFT_benchmark_Convolution_Networks/
  ./runme_chronotimer.sh
fi

# Simple test to confirm installation
#$build_dir/test/src/fft 2 12

# TODO: ADD More options to perform benchmark and testing
