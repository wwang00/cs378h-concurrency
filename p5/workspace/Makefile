PROG = nbody
CC = mpicxx
CPPFLAGS = -Wall -O3
OBJS = main.o

all : $(PROG)

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

%.o : %.c
	$(CC) $(CPPFLAGS) -c $<

.PHONY : clean
clean :
	rm -f $(PROG) $(OBJS) *.out
