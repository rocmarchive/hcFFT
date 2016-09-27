#!/bin/bash
#This script is invoked to test all generators of the hcfft library
#Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

export OPENCL_INCLUDE_PATH=$AMDAPPSDKROOT/include
export OPENCL_LIBRARY_PATH=$AMDAPPSDKROOT/lib/x86_64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCL_LIBRARY_PATH

# Move to gtest bin
working_dir1="$current_work_dir/../../build/test/unit/gtest/hcfft_Create_Destroy_Plan/bin/"
if [ ! -d "$working_dir1" ]; then
  echo "Please run script[test.sh] from hcfft/test/unit/"
  exit
fi
cd $working_dir1
rm -f $working_dir1/gtestlog.txt

#Gtest functions
unittest="$working_dir1/hcfft_Create_destroy_Plan"

runcmd1="$unittest >> gtestlog.txt"
eval $runcmd1

Log_file="$working_dir1/gtestlog.txt"
if [ ! -s "$Log_file" ]; then
  echo "${red}GTEST IS NOT WORKING....${reset}"
else
  if grep -q FAILED "$Log_file";
  then
    echo "${red}hcfft_Create_destroy_Plan               ----- [ FAILED ]${reset}"
  elif grep -q PASSED "$Log_file";
  then
    echo "${green}hcfft_Create_destroy_Plan             ----- [ PASSED ]${reset}"
    rm -f Log_file
  fi
fi

testdirectories=(hcfft_1D_transform hcfft_2D_transform hcfft_3D_transform)

while read line; do
  N1=$(echo $line | cut -f1 -d" " )
  N2=$(echo $line | cut -f2 -d" " )

## now loop through the above array
  for i in 0 1 2
  do
    working_dir1="$current_work_dir/../../build/test/unit/gtest/${testdirectories[$i]}/bin/"
    cd $working_dir1
    rm -f $working_dir1/gtestlog.txt

    #Gtest functions
    unittest="$working_dir1/${testdirectories[$i]} $N1 $N2"

    runcmd1="$unittest >> gtestlog.txt"
    eval $runcmd1

    Log_file="$working_dir1/gtestlog.txt"
    if [ ! -s "$Log_file" ]; then
      echo "${red}GTEST IS NOT WORKING....${reset}"
    else
      if grep -q FAILED "$Log_file";
      then
        echo "${red}${testdirectories[$i]} $N1 $N2             ----- [ FAILED ]${reset}"
      elif grep -q PASSED "$Log_file";
      then
        echo "${green}${testdirectories[$i]} $N1 $N2           ----- [ PASSED ]${reset}"
        rm -f $working_dir1/gtestlog.txt
      fi
    fi

    #Gtest functions
    unittest="$working_dir1/${testdirectories[$i]}_double $N1 $N2"

    runcmd1="$unittest >> gtestlog.txt"
    eval $runcmd1

    Log_file="$working_dir1/gtestlog.txt"
    if [ ! -s "$Log_file" ]; then
      echo "${red}GTEST IS NOT WORKING....${reset}"
    else
      if grep -q FAILED "$Log_file";
      then
       echo "${red}${testdirectories[$i]}_double $N1 $N2       ----- [ FAILED ]${reset}"
      elif grep -q PASSED "$Log_file";
      then
        echo "${green}${testdirectories[$i]}_double $N1 $N2    ----- [ PASSED ]${reset}"
        rm -f $working_dir1/gtestlog.txt
      fi
    fi
  done

#Input file
done < $current_work_dir/Input.txt
