PROG = pfxsum
CC = g++
CPPFLAGS = -std=c++11 -stdlib=libc++ -Wall -O3
LDFLAGS = -stdlib=libc++
OBJS = main.o argparse.o

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

main.o :
	$(CC) $(CPPFLAGS) -c main.cc

argparse.o : argparse.h
	$(CC) $(CPPFLAGS) -c argparse.cc

.PHONY : test
test:
	make
	./pfxsum < test.in > test.out

.PHONY : clean
clean :
	rm -f $(PROG) $(OBJS) *.out
