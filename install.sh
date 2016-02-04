# This script is invoked to install the hcfft library and test sources
# Preliminary version

# CHECK FOR COMPILER PATH
if [ ! -z $MCWHCCBUILD] 
then
  if [ -x $MCWHCCBUILD/compiler/bin/clang++ ] 
  then
    cmake_c_compiler=$MCWHCCBUILD/compiler/bin/clang
    cmake_cxx_compiler=$MCWHCCBUILD/compiler/bin/clang++
  fi

elif [ -x "/opt/hcc/bin/clang++" ] 
then
  cmake_c_compiler=/opt/hcc/bin/clang
  cmake_cxx_compiler=/opt/hcc/bin/clang++
else
  echo "Clang compiler not found"
  exit 1
fi

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

#Inputs converted to smallcase format
input1=$1
var1=${input1,,}
var1="test=off"

set +e
# MAKE BUILD DIR
mkdir $current_work_dir/build
mkdir $current_work_dir/build/lib
mkdir $current_work_dir/build/test
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build

#change to library build
cd $build_dir

# Cmake and make libhcfft: Install hcFFT
cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir
sudo make install

# Build Tests
cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/

set +e
mkdir $current_work_dir/build/test/examples/bin/
mkdir $current_work_dir/build/test/src/bin/
mkdir $current_work_dir/build/test/unit/gtest/bin/
set -e

make
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# KERNEL CACHE DIR
mkdir -p /tmp/kernCache

#Test=OFF (Build library and tests)
if ([ "$var1" = "test=off" ]); then
   echo "${green}HCFFT Installation Completed!${reset}"
#Test=ON (Build and test the library)
elif ([ "$var1" = "test=on" ]); then
   chmod +x $current_work_dir/test/unit/test.sh
   cd $current_work_dir/test/unit/
# Invoke test script
   ./test.sh
fi
# Simple test to confirm installation
#$build_dir/test/src/fft 2 12

# TODO: ADD More options to perform benchmark and testing
