PROGS = seq
OBJS = $(foreach prog,$(PROGS),$(prog).o) argparse.o rng.o

seq : $(OBJS)
	nvcc -o seq $(OBJS)

seq.o :
	nvcc -c seq.cu

argparse.o : argparse.h
	g++ -c argparse.cpp

rng.o : rng.h
	g++ -c rng.cpp

.PHONY : test
test :
	make
	./seq -k 16 -d 16 -i n2048-d16-c16.txt -m 150 -t 1e-5 -s 8675309

.PHONY : clean
clean :
	rm -f seq $(OBJS)
