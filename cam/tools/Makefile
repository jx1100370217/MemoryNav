# 鱼眼转柱面视角工具 - Makefile 构建（无需 CMake）
CXX = g++
CXXFLAGS = -std=c++17 -O2
LDFLAGS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)
CXXFLAGS += $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
CXXFLAGS += -I/usr/include/eigen3
LDFLAGS += -lyaml-cpp

TARGET = fisheye_to_cylindrical
SRC = fisheye_to_cylindrical.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
