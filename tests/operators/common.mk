DEFINE ?=

ROOT = ../../..

INCLUDE += -I$(ROOT)

CCFLAGS = -Wall -O3 -use_fast_math

.PHONY: all

all: $(OBJ).exe $(OBJ).dump

$(OBJ).o: $(OBJ).cpp $(ROOT)/src/operators/$(OBJ)/$(OBJ).tpp
	g++ $(CCFLAGS) $(INCLUDE) $(DEFINE) -c -o $@  $<  

$(OBJ).exe: $(OBJ).o
	g++ $(CCFLAGS) -o $@ $< $(DEFINE) $(INCLUDE)

$(OBJ).dump: $(OBJ).o
	objdump -dS $< > $@
