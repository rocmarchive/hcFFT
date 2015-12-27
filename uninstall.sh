# This script is invoked to uninstall the hcfft library and test sources
# Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

#Move to library build
cd $current_work_dir/lib/build/linux

#Invoke clean script
sh clean.sh

#Move to test build
cd $current_work_dir/test/build/linux

#Invoke clean script
sh clean.sh

#Remove cached kernel and binary objects
sudo rm -f /tmp/*.bin
sudo rm -f /tmp/libkernel*.so


# TODO: ADD More options


