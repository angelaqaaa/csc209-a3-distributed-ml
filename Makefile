
FLAGS = -Wall -Wextra -g
CC = gcc

PORT = 4242

all: server worker gen_data

server: server.c net_utils.c io_utils.c model.c protocol.h net_utils.h io_utils.h model.h
	$(CC) $(FLAGS) -DPORT=$(PORT) -o server server.c net_utils.c io_utils.c model.c -lm

worker: worker.c net_utils.c io_utils.c model.c data.c protocol.h net_utils.h io_utils.h model.h data.h
	$(CC) $(FLAGS) -DPORT=$(PORT) -o worker worker.c net_utils.c io_utils.c model.c data.c -lm

gen_data: gen_data.c
	$(CC) $(FLAGS) -o gen_data gen_data.c -lm

clean:
	rm -f server worker gen_data *.o

.PHONY: all clean
