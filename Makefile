# Adapted from https://github.com/pytorch/pytorch/blob/v2.0.0/Makefile
# This Makefile does nothing but delegating the actual building to CMake.

.PHONY: all clean

all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

test:
	@cd build && ctest .

clean:
	@rm -r build
