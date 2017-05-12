CCFLAGS = -std=c++11 -I/work/paddle
CC = nvcc

all : place_test

place.o : place.h place.cu
	$(CC) $(CCFLAGS) -c place.cu -o $@

place_test : place.o place_test.cu
	$(CC) $(CCFLAGS) -lgtest -lgtest_main $^ -o $@
