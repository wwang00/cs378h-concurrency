CC = g++
CPPFLAGS = -std=c++11 -Wall -O3
LDFLAGS = -std=c++11

all : generate

generate : generate.o
	$(CC) $(LDFLAGS) -o generate generate.o

generate.o : generate.cpp
	$(CC) $(CPPFLAGS) -c generate.cpp

.PHONY : clean
clean :
	rm -f generate generate.o
