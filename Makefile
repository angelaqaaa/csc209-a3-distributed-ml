PORT = 4242
FLAGS = -Wall -Wextra -g -DPORT=$(PORT)

all: server worker

server: server.o net_utils.o io_utils.o model.o
	gcc ${FLAGS} -o $@ $^ -lm

worker: worker.o net_utils.o io_utils.o model.o data.o
	gcc ${FLAGS} -o $@ $^ -lm

server.o: server.c protocol.h net_utils.h io_utils.h model.h
	gcc ${FLAGS} -c $<

worker.o: worker.c protocol.h net_utils.h io_utils.h model.h data.h
	gcc ${FLAGS} -c $<

net_utils.o: net_utils.c net_utils.h
	gcc ${FLAGS} -c $<

io_utils.o: io_utils.c io_utils.h protocol.h
	gcc ${FLAGS} -c $<

model.o: model.c model.h
	gcc ${FLAGS} -c $<

data.o: data.c data.h
	gcc ${FLAGS} -c $<

clean:
	rm -f *.o server worker

.PHONY: all clean