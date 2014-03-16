CC=g++
CFLAGS=
LDFLAGS=`pkg-config opencv --libs --cflags`
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=janus

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): 
	$(CC) $(SOURCES) $(CFLAGS) $(LDFLAGS) -o $@ 
