TARGET = fft
LIBS = -lampfft -L${AMPFFT_LIBRARY_PATH}
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
	-rm -f ../kernel*.cpp
	-rm -f ../libFFTKernel*.so
