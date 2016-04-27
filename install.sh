# This script is invoked to install the hcfft library and test sources
# Preliminary version

# CHECK FOR COMPILER PATH
if [ ! -z $MCWHCCBUILD ]
then
  if [ -x $MCWHCCBUILD/compiler/bin/clang++ ] 
  then
    cmake_c_compiler=$MCWHCCBUILD/compiler/bin/clang
    cmake_cxx_compiler=$MCWHCCBUILD/compiler/bin/clang++
  fi

elif [ -x "/opt/rocm/hcc/bin/clang++" ]
then
  cmake_c_compiler=/opt/rocm/hcc/bin/clang
  cmake_cxx_compiler=/opt/rocm/hcc/bin/clang++
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

  1) ${green}--path${reset}    Path to your hcfft installation.(default path is /opt/rocm/ - needs sudo access)
  2) ${green}--test${reset}    Test to enable the library testing.
  3) ${green}--bench${reset}   Profile benchmark using chrono timer.
===================================================================================================================
Usage: ./install.sh --path=/path/to/user/installation --test=on
===================================================================================================================
Example:
(1) ${green}./install.sh --path=/path/to/user/installation --test=on
       <library gets installed in /path/to/user/installation, testing = on>
(2) ${green}./install.sh --test=on${reset} (sudo access needed)
       <library gets installed in /opt/rocm/, testing = on>
(3) ${green}./install.sh --bench=on
       <library gets installed in /opt/rocm/, bench = on>

===================================================================================================================
HELP
exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --path=*)
      path="${1#*=}"
      ;;
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

if [ -z $path ]; then
    path="/opt/rocm/"
fi

if [ "$path" = "/opt/rocm/" ]; then
   set +e
   sudo mkdir -p /opt/rocm/
   set -e
fi

export hcfft_install=$path
set hcfft_install=$path

export OPENCL_INCLUDE_PATH=$AMDAPPSDKROOT/include
export OPENCL_LIBRARY_PATH=$AMDAPPSDKROOT/lib/x86_64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCL_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$hcfft_installlib/hcfft
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$hcfft_install/include/hcfft

set +e
# MAKE BUILD DIR
mkdir -p $current_work_dir/build
mkdir -p $current_work_dir/build/lib
mkdir -p $current_work_dir/build/test
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build

#change to library build
cd $build_dir

# Cmake and make libhcfft: Install hcFFT
cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir
if [ "$path" = "/opt/rocm/" ]; then
    sudo make install
else
    make install
fi

# Build Tests
cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/

set +e
mkdir -p $current_work_dir/build/test/examples/bin/
mkdir -p $current_work_dir/build/test/src/bin/
mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_1D_transform/bin/
mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_2D_transform/bin/
mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_3D_transform/bin/
mkdir -p $current_work_dir/build/test/unit/gtest/hcfft_Create_Destroy_Plan/bin/
set -e

make
#red=`tput setaf 1`
#green=`tput setaf 2`
#reset=`tput sgr0`

# KERNEL CACHE DIR
mkdir -p $HOME/kernCache

#Test=OFF (Build library and tests)
if ( [ -z $testing ] ) || ( [ "$testing" = "off" ] ); then
  echo "${green}HCFFT Installation Completed!${reset}"
# Test=ON (Build and test the library)
elif ( [ "$testing" = "on" ] ); then
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

if grep --quiet hcfft ~/.bashrc; then
  cd $current_work_dir
else
  eval "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> ~/.bashrc"
  cd $current_work_dir
  exec bash
fi

# TODO: ADD More options to perform benchmark and testing
