PROG = kalman_seq
CC = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS = -lcblas
NVCC = nvcc -arch=sm_61
OBJS = kalman_seq.o argparse.o

all : $(PROG)

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

%.o : %.cpp %.h
	$(CC) $(CXXFLAGS) -c $<

.PHONY : clean
clean :
	rm -f $(PROG) $(OBJS)
