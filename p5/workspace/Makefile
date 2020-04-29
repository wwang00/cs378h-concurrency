PROG = nbody
CC = mpicxx
CPPFLAGS = -std=c++11 -Wall -O3
LDFLAGS = -std=c++11
OBJS = main.o argparse.o datatypes.o globals.o tree.o

all : $(PROG)

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

main.o : main.cpp
	$(CC) $(CPPFLAGS) -c main.cpp

%.o : %.cpp %.h
	$(CC) $(CPPFLAGS) -c $<

.PHONY : clean
clean :
	rm -f $(PROG) $(OBJS) *.out
