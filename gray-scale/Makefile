CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 $(shell pkg-config --cflags opencv4)
LDLIBS = $(shell pkg-config --libs opencv4)

TARGET = image_viewer
SRC = main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

run:
	./$(TARGET) assets/san-viecente-de-la-sonsierra.jpg

clean:
	rm -f $(TARGET)
