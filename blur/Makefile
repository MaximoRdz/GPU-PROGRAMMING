CXX = g++
NVCC = nvcc

CXXFLAGS = -Wall -Wextra -std=c++17 $(shell pkg-config --cflags opencv4)
LDLIBS = $(shell pkg-config --libs opencv4) -lcudart -L$(CUDA_LIB_DIR)

CUDA_PATH = /usr/local/cuda

CUDA_LIB_DIR = $(CUDA_PATH)/lib64
NVCCFLAGS = -std=c++17 -O2 -arch=native

TARGET = image_viewer
CPP_SRC = main.cpp
CU_SRC = blur_cuda.cu
CU_OBJ = blur_cuda.o

$(TARGET): $(CU_OBJ) $(CPP_SRC) blur_cuda.h
	$(CXX) $(CXXFLAGS) $(CPP_SRC) $(CU_OBJ) -o $(TARGET) $(LDLIBS)

$(CU_OBJ): $(CU_SRC) blur_cuda.h
	$(NVCC) $(NVCCFLAGS) -c $(CU_SRC) -o $(CU_OBJ)

run:
	./$(TARGET) ../assets/san-vicente-de-la-sonsierra.jpg

clean:
	rm -f $(TARGET) $(CU_OBJ)
