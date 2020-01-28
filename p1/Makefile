PROG = pfxsum
CC = g++
CPPFLAGS = -std=c++11 -Wall -O3
LDFLAGS = -stdlib=libc++
OBJS = main.o argparse.o scan.o

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

main.o :
	$(CC) $(CPPFLAGS) -c main.cc

argparse.o : argparse.h
	$(CC) $(CPPFLAGS) -c argparse.cc

scan.o : scan.h
	$(CC) $(CPPFLAGS) -c scan.cc

.PHONY : test
test:
	make
	./pfxsum < test.in > test.out

.PHONY : clean
clean :
	rm -f $(PROG) $(OBJS) *.out
