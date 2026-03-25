#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

/* Message type codes */
#define MSG_REGISTER   0x01
#define MSG_WEIGHTS    0x02
#define MSG_GRADIENT   0x03
#define MSG_DONE       0x04

/* Header size: 1 byte type + 4 bytes payload_size */
#define HEADER_SIZE    5

/* Configuration */
#define MAX_WORKERS      8
#define MAX_FEATURES    64
#define MAX_ROUNDS     100
#define LEARNING_RATE  0.01f
#define LOSS_THRESHOLD 0.001f

/* Default port — overridden by Makefile -DPORT */
#ifndef PORT
#define PORT 4242
#endif

/* Buffer large enough for any single message */
#define RECV_BUF_SIZE  (HEADER_SIZE + 12 + MAX_FEATURES * sizeof(float))

/* Per-worker tracking on the server side */
struct worker_info {
    int fd;                          /* -1 if slot empty */
    int state;                       /* 0=new, 1=registered, 2=sent_weights */
    int num_samples;
    char recv_buf[RECV_BUF_SIZE];    /* accumulation buffer for partial reads */
    int recv_len;
    float gradients[MAX_FEATURES];
    float local_loss;
    int gradient_received;           /* 1 = received for current round */
};

#endif
