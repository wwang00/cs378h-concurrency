PROG = pfxsum
CC = g++
CPPFLAGS = -std=c++11 -Wall -O3
LDFLAGS = -std=c++11 -pthread
OBJS = main.o argparse.o scan.o

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

main.o :
	$(CC) $(CPPFLAGS) -c main.cpp

scan.o : scan.h
	$(CC) $(CPPFLAGS) -c scan.cpp

argparse.o : argparse.h
	$(CC) $(CPPFLAGS) -c argparse.cpp

.PHONY : test
test:
	make
	./pfxsum -n $(n) -i $(f).txt -o $(f).out

.PHONY : clean
clean :
	rm -f $(PROG) $(OBJS) *.out
