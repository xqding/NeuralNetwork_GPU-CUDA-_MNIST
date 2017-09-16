CC = nvcc -std=c++11
ARCH=-arch sm_20
NVOPTS=-O3 $(ARCH) -x cu
LIBS = -lcublas

BUILD = ./build
SOURCE = ./src

programs = $(BUILD)/main
objects = $(BUILD)/NN.o $(BUILD)/kernels.o $(BUILD)/functions.o

all: $(programs) copy

$(BUILD)/%.o: $(SOURCE)/%.cpp $(SOURCE)/%.h
	$(CC) $(NVOPTS) -c $< -o $@

$(BUILD)/%.o: $(SOURCE)/%.cpp
	$(CC) $(NVOPTS) -c $< -o $@

$(programs): %: %.o $(objects)
	$(CC) $(LIBS) $(objects) $< -o $@

copy:
	cp $(programs) ./test/

.PHONY: clean copy
clean:
	rm -rf $(objects) $(programs) $(BUILD)/main.o *~
