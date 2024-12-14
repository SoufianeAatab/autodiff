CC=g++

CCFLAGS=-O3 

OBJS=conv maxpool test sockets

.PHONY: all clean

%.o: %.cc
	$(CC) $(CCFLAGS) -c $@ $< 

%: %.o
	$(CC) $(CCFLAGS) -o $@ $<

all: $(OBJS)

clean:
	rm $(OBJS)