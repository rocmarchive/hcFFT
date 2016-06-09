# This script is invoked to uninstall the hcfft library and test sources
# Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

# Remove build
rm -rf $current_work_dir/build

# Remove temporarly cached kernel shared objects and binaries
rm -rf $HOME/kernCache

# TODO: ADD More options


