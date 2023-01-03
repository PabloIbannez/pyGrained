INCLUDE := $(shell python3 -m pybind11 --includes) 
EXTENSION := $(shell python3-config --extension-suffix)

cmptK:
	 g++ -O3 -Wall -shared -std=c++11 -fPIC $(INCLUDE) computeK.cpp -o computeK$(EXTENSION)
