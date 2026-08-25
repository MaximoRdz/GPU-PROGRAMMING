CXX  := g++
NVCC := nvcc

TARGET := app

CUDA_PATH := /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_PATH)/lib64

CXXFLAGS := -Wall -Wextra -std=c++17 -O2 \
            -Iinclude \
			-I$(CUDA_PATH)/include \
            $(shell pkg-config --cflags opencv4) \
            -MMD -MP

NVCCFLAGS := -std=c++17 -O2 \
             -Iinclude \
             -MMD -MP

LDFLAGS := -L$(CUDA_LIB_DIR)

LDLIBS := $(shell pkg-config --libs opencv4) -lcudart


CPP_SRC := $(wildcard src/*.cpp) \
           $(wildcard benchmarks/*.cpp)

CU_SRC := $(wildcard src/*.cu) \
          $(wildcard benchmarks/*.cu)

CPP_OBJ := $(CPP_SRC:%.cpp=build/cpp/%.o)
CU_OBJ  := $(CU_SRC:%.cu=build/cuda/%.o)

OBJ := $(CPP_OBJ) $(CU_OBJ)
DEP := $(OBJ:.o=.d)


$(TARGET): $(OBJ)
	$(NVCC) $^ -o $@ $(LDFLAGS) $(LDLIBS)


build/cpp/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@


build/cuda/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


run: $(TARGET)
	./$(TARGET)


clean:
	rm -rf build $(TARGET)


-include $(DEP)


.PHONY: run clean
