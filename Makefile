.PHONY: all clean

all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

clean:
	@rm -r build
