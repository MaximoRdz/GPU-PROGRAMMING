CXX = g++
NVCC = nvcc

CUDA_PATH = /usr/local/cuda
CUDA_LIB_DIR = $(CUDA_PATH)/lib64

CXXFLAGS = -Wall -Wextra -std=c++17 $(shell pkg-config --cflags opencv4) -MMD -MP
LDLIBS = $(shell pkg-config --libs opencv4) -lcudart -L$(CUDA_LIB_DIR)

NVCCFLAGS = -std=c++17 -O2 -arch=native

TARGET = image_viewer

CPP_SRC := $(wildcard *.cpp)
C_SRC   := $(wildcard *.c)
CU_SRC  := blur_cuda.cu

CPP_OBJ := $(CPP_SRC:.cpp=.o)
C_OBJ   := $(C_SRC:.c=.o)
CU_OBJ  := $(CU_SRC:.cu=.o)

OBJ := $(CPP_OBJ) $(C_OBJ) $(CU_OBJ)
DEP := $(OBJ:.o=.d)


$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


run: $(TARGET)
	./$(TARGET) ../assets/san-vicente-de-la-sonsierra.jpg


clean:
	rm -f $(TARGET) $(OBJ) $(DEP)


-include $(DEP)
