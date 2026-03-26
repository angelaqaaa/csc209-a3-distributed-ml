FLAGS = -Wall -Wextra -g
CC = gcc

PORT ?= 4242

all: server worker

server: server.c net_utils.c io_utils.c model.c protocol.h net_utils.h io_utils.h model.h
	$(CC) $(FLAGS) -DPORT=$(PORT) -o server server.c net_utils.c io_utils.c model.c

worker: worker.c net_utils.c io_utils.c model.c data.c protocol.h net_utils.h io_utils.h model.h data.h
	$(CC) $(FLAGS) -DPORT=$(PORT) -o worker worker.c net_utils.c io_utils.c model.c data.c

clean:
	rm -f server worker *.o

.PHONY: all clean
