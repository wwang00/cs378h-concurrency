CC = g++ -O3
NVCC = nvcc -arch=sm_60
OBJS = argparse.o rng.o

N = 2048
DIMS = 16
INPUT = ../input/n$(N)-d$(DIMS).txt

seq : seq.o $(OBJS)
	$(CC) -o seq seq.o $(OBJS)

thrust : thrust.o $(OBJS)
	$(NVCC) -o thrust thrust.o $(OBJS)

cuda : cuda.o $(OBJS)
	$(NVCC) -o cuda cuda.o $(OBJS)

seq.o :
	$(CC) -c seq.cpp

thrust.o:
	$(NVCC) -c thrust.cu

cuda.o:
	$(NVCC) -c cuda.cu

argparse.o : argparse.h
	$(CC) -c argparse.cpp

rng.o : rng.h
	$(CC) -c rng.cpp

.PHONY : seqtest
seqtest :
	make seq
	./seq -k 16 -d $(DIMS) -i $(INPUT) -m 150 -t 1e-5 -s 8675309 -c

.PHONY : thrusttest
thrusttest :
	make thrust
	./thrust -k 16 -d $(DIMS) -i $(INPUT) -m 150 -t 1e-5 -s 8675309 -c

.PHONY : cudatest
cudatest :
	make cuda
	./cuda -k 16 -d $(DIMS) -i $(INPUT) -m 150 -t 1e-5 -s 8675309 -c

.PHONY : clean
clean :
	rm -f seq thrust cuda seq.o thrust.o cuda.o $(OBJS)
