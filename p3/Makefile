NVCC = nvcc -arch=sm_60
OBJS = argparse.o rng.o

seq : seq.o $(OBJS)
	$(NVCC) -o seq seq.o $(OBJS)

thrust : thrust.o $(OBJS)
	$(NVCC) -o thrust thrust.o $(OBJS)

seq.o :
	$(NVCC) -c seq.cu

thrust.o:
	$(NVCC) -c thrust.cu

argparse.o : argparse.h
	g++ -c argparse.cpp

rng.o : rng.h
	g++ -c rng.cpp

.PHONY : seqtest
seqtest :
	make seq
	./seq -k 16 -d 16 -i n2048-d16-c16.txt -m 150 -t 1e-5 -s 8675309 -c

.PHONY : thrusttest
thrusttest :
	make thrust
	./thrust -k 16 -d 16 -i n2048-d16-c16.txt -m 150 -t 1e-5 -s 8675309 -c

.PHONY : clean
clean :
	rm -f seq thrust seq.o thrust.o $(OBJS)
