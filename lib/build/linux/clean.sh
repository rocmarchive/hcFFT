# Script to clean up library generated files

# Clean local cmakes and make
rm -rf CMake* Makefile cmake*

# Clean up locally generated library shared object
rm libhcfft.so

# Clean up system wide installed headers and shared objects
sudo xargs rm < install_manifest.txt

# Remove install log
rm install_manifest.txt
