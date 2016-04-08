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

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# Help menu
print_help() {
cat <<-HELP
===================================================================================================================
This script is invoked to install hcFFT library and test sources. Please provide the following arguments:

  1) ${green}--path${reset}    Path to your hcfft installation.(default path is /opt/ROCm/ - needs sudo access)
  2) ${green}--test${reset}    Test to enable the library testing.

===================================================================================================================
Usage: ./install.sh --path=/path/to/user/installation --test=on
===================================================================================================================
Example:
(1) ${green}./install.sh --path=/path/to/user/installation --test=on
       <library gets installed in /path/to/user/installation, testing = on>
(2) ${green}./install.sh --test=on${reset} (sudo access needed)
       <library gets installed in /opt/ROCm/, testing = on>

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
    --help) print_help;;
    *)
      printf "************************************************************\n"
      printf "* Error: Invalid arguments, run --help for valid arguments.*\n"
      printf "************************************************************\n"
      exit 1
  esac
  shift
done

if [ -z $path ]; then
    path="/opt/ROCm/"
fi

if [ "$path" = "/opt/ROCm/" ]; then
   set +e
   sudo mkdir /opt/ROCm/
   set -e
fi

export hcfft_install=$path
set hcfft_install=$path

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$hcfft_install/lib/hcfft
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$hcfft_install/include/hcfft

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
if [ "$path" = "/opt/ROCm/" ]; then
    sudo make install
else
    make install
fi

# Build Tests
cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/

set +e
mkdir $current_work_dir/build/test/examples/bin/
mkdir $current_work_dir/build/test/src/bin/
mkdir $current_work_dir/build/test/unit/gtest/hcfft_1D_transform/bin/
mkdir $current_work_dir/build/test/unit/gtest/hcfft_2D_transform/bin/
mkdir $current_work_dir/build/test/unit/gtest/hcfft_3D_transform/bin/
mkdir $current_work_dir/build/test/unit/gtest/hcfft_Create_Destroy_Plan/bin/
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

# Simple test to confirm installation
#$build_dir/test/src/fft 2 12

if grep --quiet hcfft ~/.bashrc; then
  echo
else
  eval "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> ~/.bashrc"
fi

cd $current_work_dir
exec bash
# TODO: ADD More options to perform benchmark and testing
