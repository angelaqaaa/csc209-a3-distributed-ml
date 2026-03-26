#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include "io_utils.h"

/*
 * Read exactly nbytes from fd into buf.
 * Returns 0 on success, -1 on error or disconnect.
 * Used by worker (blocking — only talks to one server).
 */
int read_all(int fd, void *buf, size_t nbytes) {
    size_t total = 0;
    while (total < nbytes) {
        int n = read(fd, (char *)buf + total, nbytes - total);
        if (n <= 0) {
            return -1;
        }
        total += n;
    }
    return 0;
}

/*
 * Write exactly nbytes from buf to fd.
 * Returns 0 on success, -1 on error.
 */
int write_all(int fd, const void *buf, size_t nbytes) {
    size_t total = 0;
    while (total < nbytes) {
        int n = write(fd, (const char *)buf + total, nbytes - total);
        if (n <= 0) {
            return -1;
        }
        total += n;
    }
    return 0;
}

/*
 * Read available bytes from worker's fd into its recv_buf.
 * Does exactly ONE read() call — never blocks on a single client.
 * Returns:  1 if a complete message is assembled
 *           0 if message is still incomplete
 *          -1 if worker disconnected or error
 */
int accumulate_read(struct worker_info *w) {
    int nbytes = read(w->fd, w->recv_buf + w->recv_len,
                      RECV_BUF_SIZE - w->recv_len);
    if (nbytes <= 0) {
        return -1;
    }
    w->recv_len += nbytes;

    /* Need at least the header to know message size */
    if (w->recv_len < HEADER_SIZE) {
        return 0;
    }

    /* Parse payload_size from header bytes 1-4 */
    uint32_t payload_size;
    memcpy(&payload_size, w->recv_buf + 1, 4);
    payload_size = ntohl(payload_size);

    int total_needed = HEADER_SIZE + (int)payload_size;
    if (w->recv_len < total_needed) {
        return 0;
    }
    return 1;
}
