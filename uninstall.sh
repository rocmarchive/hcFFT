# This script is invoked to uninstall the hcfft library and test sources
# Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

# Remove system wide installed lib and headers
sudo xargs rm < $current_work_dir/build/lib/install_manifest.txt

# Remove build
sudo rm -rf $current_work_dir/build

# TODO: ADD More options


