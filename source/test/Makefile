TARGET = fft
LIBS = -lhcfft -L${HCFFT_LIBRARY_PATH}
CC = ${MCWCPPAMPROOT}/build/compiler/bin/clang++
CFLAGS = `${MCWCPPAMPROOT}/build/build/Release/bin/clamp-config --build --cxxflags --ldflags`

.PHONY: default all clean 

default: $(TARGET)
all: default

.PRECIOUS: $(TARGET) 


$(TARGET):fft_test.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) $< -o $@
			
clean:
	-rm -f $(TARGET)
	-rm -f fft_test.o
