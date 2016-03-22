#!/bin/bash
#This script is invoked to test all generators of the hcfft library
#Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# Move to gtest bin
working_dir1="$current_work_dir/../../build/test/unit/gtest/hcfft_Create_Destroy_Plan/bin/"
cd $working_dir1

#Gtest functions
unittest="$working_dir1/hcfft_Create_destroy_Plan"

runcmd1="$unittest >> gtestlog.txt"
eval $runcmd1

Log_file="$working_dir1/gtestlog.txt"
if grep -q FAILED "$Log_file";
then
    echo "${red}hcfft_Create_destroy_Plan               ----- [ FAILED ]${reset}"
else
    echo "${green}hcfft_Create_destroy_Plan             ----- [ PASSED ]${reset}"
    rm $working_dir1/gtestlog.txt
fi

testdirectories=(hcfft_1D_transform hcfft_2D_transform hcfft_3D_transform)

## now loop through the above array
for i in 0 1 2
do
  working_dir1="$current_work_dir/../../build/test/unit/gtest/${testdirectories[$i]}/bin/"
  cd $working_dir1

  #Gtest functions
  unittest="$working_dir1/${testdirectories[$i]}"

  runcmd1="$unittest >> gtestlog.txt"
  eval $runcmd1

  Log_file="$working_dir1/gtestlog.txt"
  if grep -q FAILED "$Log_file";
  then
      echo "${red}${testdirectories[$i]}               ----- [ FAILED ]${reset}"
  else
      echo "${green}${testdirectories[$i]}             ----- [ PASSED ]${reset}"
      rm $working_dir1/gtestlog.txt
  fi

  #Gtest functions
  unittest="$working_dir1/${testdirectories[$i]}_double"

  runcmd1="$unittest >> gtestlog.txt"
  eval $runcmd1

  Log_file="$working_dir1/gtestlog.txt"
  if grep -q FAILED "$Log_file";
  then
      echo "${red}${testdirectories[$i]}_double               ----- [ FAILED ]${reset}"
  else
      echo "${green}${testdirectories[$i]}_double           ----- [ PASSED ]${reset}"
      rm $working_dir1/gtestlog.txt
  fi

  #Gtest functions
  unittest="$working_dir1/${testdirectories[$i]}_padding"

  runcmd1="$unittest >> gtestlog.txt"
  eval $runcmd1

  Log_file="$working_dir1/gtestlog.txt"
  if grep -q FAILED "$Log_file";
  then
      echo "${red}${testdirectories[$i]}_padding               ----- [ FAILED ]${reset}"
  else
      echo "${green}${testdirectories[$i]}_padding           ----- [ PASSED ]${reset}"
      rm $working_dir1/gtestlog.txt
  fi
done
