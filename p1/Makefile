PROG = pfxsum
CC = g++
CPPFLAGS = -std=c++11 -Wall -O3
LDFLAGS = -stdlib=libc++
OBJS = main.o argparse.o scan_ops.o

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

main.o :
	$(CC) $(CPPFLAGS) -c main.cpp

argparse.o : argparse.h
	$(CC) $(CPPFLAGS) -c argparse.cpp

scan_ops.o : scan_ops.h
	$(CC) $(CPPFLAGS) -c scan_ops.cpp

.PHONY : test
test:
	make
	./pfxsum < test.in > test.out

.PHONY : clean
clean :
	rm -f $(PROG) $(OBJS) *.out
