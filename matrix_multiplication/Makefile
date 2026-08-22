CXX  := g++
NVCC := nvcc

TARGET := app

CUDA_PATH := /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_PATH)/lib64

CXXFLAGS := -Wall -Wextra -std=c++17 -O2 \
            -Iinclude \
            $(shell pkg-config --cflags opencv4) \
            -MMD -MP

NVCCFLAGS := -std=c++17 -O2 \
             -Iinclude

LDFLAGS := -L$(CUDA_LIB_DIR)

LDLIBS := $(shell pkg-config --libs opencv4) -lcudart

CPP_SRC := $(wildcard src/*.cpp)
CU_SRC  := $(wildcard src/*.cu)

CPP_OBJ := $(patsubst src/%.cpp,build/%.o,$(CPP_SRC))
CU_OBJ  := $(patsubst src/%.cu,build/%.o,$(CU_SRC))

OBJ := $(CPP_OBJ) $(CU_OBJ)
DEP := $(OBJ:.o=.d)


$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)


build/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@


build/%.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@


run: $(TARGET)


clean:
	rm -rf build $(TARGET)


-include $(DEP)


.PHONY: run clean
