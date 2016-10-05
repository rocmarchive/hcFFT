#!/bin/bash
#This script is invoked to test all generators of the hcfft library
#Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

export HCFFT_LIBRARY_PATH=$current_work_dir/../../build/lib/src/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HCFFT_LIBRARY_PATH

testdirectories=(hipfft_1D_transform hipfft_2D_transform hipfft_3D_transform)
test1D=(1D_C2C 1D_C2R 1D_D2Z 1D_R2C 1D_Z2D 1D_Z2Z)
test2D=(2D_C2C 2D_C2R 2D_D2Z 2D_R2C 2D_Z2D 2D_Z2Z)
test3D=(3D_C2C 3D_C2R 3D_D2Z 3D_R2C 3D_Z2D 3D_Z2Z)


## now loop through the above array
for i in 0 1 2
do
  working_dir1="$current_work_dir/../../build/test/HIP_Unit_Tests/${testdirectories[$i]}/bin/"
  cd $working_dir1

  for j in 0 1 2 3 4 5
  do
    #rm -f $working_dir1/testlog.txt
    if [ $i == 0 ];then
      unittest="$working_dir1/${test1D[$j]}"
      runcmd1="$unittest 2> testlog.txt"
      eval $runcmd1

      Log_file="$working_dir1/testlog.txt"
      if [ ! -e "$Log_file" ];then
        echo "{red}TEST IS NOT WORKING....${reset}"
      else
        if grep -q error "$Log_file";
        then
          echo "${red} HIP${test1D[$j]}               ----- [ FAILED ]${reset}"
          rm -f $working_dir1/testlog.txt
        else
          echo "${green} HIP${test1D[$j]}             ----- [ PASSED ]${reset}"
          rm -f $working_dir1/testlog.txt
        fi
      fi
    elif [ $i == 1 ];then
      unittest="$working_dir1/${test2D[$j]}"
      runcmd1="$unittest 2> testlog.txt"
      eval $runcmd1

      Log_file="$working_dir1/testlog.txt"
      if [ ! -e "$Log_file" ];then
        echo "{red}TEST IS NOT WORKING....${reset}"
      else
        if grep -q error "$Log_file";
        then
          echo "${red} HIP${test2D[$j]}               ----- [ FAILED ]${reset}"
          rm -f $working_dir1/testlog.txt
        else
          echo "${green} HIP${test2D[$j]}             ----- [ PASSED ]${reset}"
          rm -f $working_dir1/testlog.txt
        fi
      fi
    elif [ $i == 2 ];then
          unittest="$working_dir1/${test3D[$j]}"
      runcmd1="$unittest 2> testlog.txt"
      eval $runcmd1

      Log_file="$working_dir1/testlog.txt"
      if [ ! -e "$Log_file" ];then
        echo "{red}TEST IS NOT WORKING....${reset}"
      else
        if grep -q error "$Log_file";
        then
          echo "${red} HIP${test3D[$j]}               ----- [ FAILED ]${reset}"
          rm -f $working_dir1/testlog.txt
        else
          echo "${green} HIP${test3D[$j]}             ----- [ PASSED ]${reset}"
          rm -f $working_dir1/testlog.txt
        fi
      fi
    fi
  done
done
