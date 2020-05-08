TGTS = kalman_seq backtest_cuda kalman_cuda
TGT_OBJS = kalman_seq.o kalman_cuda.o backtest_cuda.o kalman.o
CC = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS = -lcblas
NVCC = nvcc -arch=sm_61
OBJS = argparse.o

all : $(TGTS)

kalman_seq : kalman_seq.o kalman.o $(OBJS)
	$(CC) $(LDFLAGS) -o kalman_seq kalman_seq.o kalman.o $(OBJS)

kalman_cuda : kalman_cuda.o $(OBJS)
	$(NVCC) -o kalman_cuda kalman_cuda.o $(OBJS)

kalman_cuda.o : kalman_cuda.cu
	$(NVCC) -c kalman_cuda.cu

backtest_cuda : backtest_cuda.o $(OBJS)
	$(NVCC) -o backtest_cuda backtest_cuda.o $(OBJS)

backtest_cuda.o : backtest_cuda.cu
	$(NVCC) -c backtest_cuda.cu

%.o : %.cpp
	$(CC) $(CXXFLAGS) -c $<

.PHONY : clean
clean :
	rm -f $(TGTS) $(TGT_OBJS) $(OBJS)
