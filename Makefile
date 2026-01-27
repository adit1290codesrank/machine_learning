CXX := g++
CXXFLAGS = -O3 -march=native -fopenmp -Wall -std=c++17 -I include

SRC_DIRS := src src/core src/layers src/io

SRCS := $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))

OBJS := $(SRCS:.cpp=.o)

TARGET := main

all :$(TARGET)

$(TARGET) : $(OBJS)
	@echo "Linking with OpenMP"
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)
	@echo "Done. Run ./$(TARGET) to execute."

%.o : %.cpp
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	@echo "Cleaning up"
	del /Q $(TARGET).exe
	del /Q src\*.o
	del /Q src\core\*.o
	del /Q src\layers\*.o
	del /Q src\io\*.o
	@echo "Done."