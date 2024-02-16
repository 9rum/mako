# Adapted from https://github.com/pytorch/pytorch/blob/v2.0.0/Makefile
# This Makefile does nothing but delegating the actual building to CMake.

.PHONY: all test clean

all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

test:
	@ctest --test-dir build

clean:
	@rm -r build
