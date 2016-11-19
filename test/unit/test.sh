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
working_dir1="$current_work_dir/../../build/test/unit/hcfft_Create_Destroy_Plan/bin"
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

test_1d_transforms=(hcfft_1D_transform hcfft_1D_transform_double)

while read line; do
  N1=$(echo $line | cut -f1 -d" " )

numtests=${#test_1d_transforms[@]}

## now loop through the above array
  for (( i=0; i<numtests; i++ ));  
  do
    working_dir1="$current_work_dir/../../build/test/unit/hcfft_transforms/bin"
    cd $working_dir1
    if [ ! -d "errlog" ]; then
      mkdir "errlog"
    fi
    errlogdir="${working_dir1}/errlog"

    #Gtest functions
    unittest="${working_dir1}/${test_1d_transforms[$i]} $N1"

    runcmd1="$unittest >> gtestlog.txt"
    eval $runcmd1

    Log_file="$working_dir1/gtestlog.txt"
    if [ ! -s "$Log_file" ]; then
      echo "${red}GTEST IS NOT WORKING....${reset}"
    else
      if grep -q FAILED "$Log_file";
      then
        echo "${red}${test_1d_transforms[$i]} $N1            ----- [ FAILED ]${reset}"
        mv "${working_dir1}/gtestlog.txt" "${errlogdir}/${test_1d_transforms[$i]}_${N1}.txt" 
      elif grep -q PASSED "$Log_file";
      then
        echo "${green}${test_1d_transforms[$i]} $N1         ----- [ PASSED ]${reset}"
        rm -f $working_dir1/gtestlog.txt
      fi
    fi
  done

#Input file
done < $current_work_dir/Input1D.txt


test_2d_transforms=(hcfft_2D_transform hcfft_2D_transform_double)

while read line; do
  N1=$(echo $line | cut -f1 -d" " )
  N2=$(echo $line | cut -f2 -d" " )

numtests=${#test_2d_transforms[@]}

## now loop through the above array
  for (( i=0; i<numtests; i++ ));  
  do
    working_dir1="$current_work_dir/../../build/test/unit/hcfft_transforms/bin/"
    cd $working_dir1
    if [ ! -d "errlog" ]; then
      mkdir "errlog"
    fi
    errlogdir="${working_dir1}/errlog"

    #Gtest functions
    unittest="${working_dir1}/${test_2d_transforms[$i]} $N1 $N2"

    runcmd1="$unittest >> gtestlog.txt"
    eval $runcmd1

    Log_file="$working_dir1/gtestlog.txt"
    if [ ! -s "$Log_file" ]; then
      echo "${red}GTEST IS NOT WORKING....${reset}"
    else
      if grep -q FAILED "$Log_file";
      then
        echo "${red}${test_2d_transforms[$i]} $N1 $N2            ----- [ FAILED ]${reset}"
        mv "${working_dir1}/gtestlog.txt" "${errlogdir}/${test_2d_transforms[$i]}_${N1}_${N2}.txt" 
      elif grep -q PASSED "$Log_file";
      then
        echo "${green}${test_2d_transforms[$i]} $N1 $N2          ----- [ PASSED ]${reset}"
        rm -f $working_dir1/gtestlog.txt
      fi
    fi
  done

#Input file
done < $current_work_dir/Input2D.txt

test_3d_transforms=(hcfft_3D_transform hcfft_3D_transform_double)

while read line; do
  N1=$(echo $line | cut -f1 -d" " )
  N2=$(echo $line | cut -f2 -d" " )
  N3=$(echo $line | cut -f3 -d" " )

numtests=${#test_2d_transforms[@]}

## now loop through the above array
  for (( i=0; i<numtests; i++ ));  
  do
    working_dir1="$current_work_dir/../../build/test/unit/hcfft_transforms/bin/"
    cd $working_dir1
    if [ ! -d "errlog" ]; then
      mkdir "errlog"
    fi
    errlogdir="${working_dir1}/errlog"

    #Gtest functions
    unittest="${working_dir1}/${test_3d_transforms[$i]} $N1 $N2 $N3"

    runcmd1="$unittest >> gtestlog.txt"
    eval $runcmd1

    Log_file="$working_dir1/gtestlog.txt"
    if [ ! -s "$Log_file" ]; then
      echo "${red}GTEST IS NOT WORKING....${reset}"
    else
      if grep -q FAILED "$Log_file";
      then
        echo "${red}${test_3d_transforms[$i]} $N1 $N2  $N3          ----- [ FAILED ]${reset}"
        mv "${working_dir1}/gtestlog.txt" "${errlogdir}/${test_3d_transforms[$i]}_${N1}_${N2}_${N3}.txt" 
      elif grep -q PASSED "$Log_file";
      then
        echo "${green}${test_3d_transforms[$i]} $N1 $N2  $N3        ----- [ PASSED ]${reset}"
        rm -f $working_dir1/gtestlog.txt
      fi
    fi
  done

#Input file
done < $current_work_dir/Input3D.txt
