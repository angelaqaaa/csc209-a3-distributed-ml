#ifndef IO_UTILS_H
#define IO_UTILS_H

#include "protocol.h"

/*
 * Read exactly nbytes from fd into buf.
 * Returns 0 on success, -1 on error or disconnect.
 * Used by worker (blocking — only talks to one server).
 */
int read_all(int fd, void *buf, size_t nbytes);

/*
 * Write exactly nbytes from buf to fd.
 * Returns 0 on success, -1 on error.
 */
int write_all(int fd, const void *buf, size_t nbytes);

/*
 * Read available bytes from worker's fd into its recv_buf.
 * Does exactly ONE read() call — never blocks on a single client.
 * Returns:  1 if a complete message is assembled
 *           0 if message is still incomplete
 *          -1 if worker disconnected or error
 */
int accumulate_read(struct worker_info *w);

/*
 * Writes a 5-byte header: 1 byte type + 4 bytes payload_len (network order).
 * Returns 0 on success, -1 on failure.
 */
int send_header(int fd, uint8_t type, uint32_t payload_len);

/*
 * Parses a 5-byte header from buf into type and payload_len (host order).
 */
void read_header(const uint8_t *buf, uint8_t *type, uint32_t *payload_len);

#endif
