DEFINE ?=

ROOT = ../../..

INCLUDE += -I$(ROOT)

CCFLAGS = -O3 -use_fast_math

$(OBJ).exe: $(OBJ).cpp $(ROOT)/src/operators/$(OBJ)/$(OBJ).tpp
	g++ $(CCFLAGS) -o $@ $< $(DEFINE) $(INCLUDE)

all: $(OBJ).exe