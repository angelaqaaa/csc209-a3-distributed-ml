#ifndef NET_UTILS_H
#define NET_UTILS_H

#include <netinet/in.h>

/* Network helper functions (Partner 2) */

/*
 * Fill a sockaddr_in structure for binding and listening on the given port.
 *   addr - pointer to sockaddr_in to fill
 *   port - TCP port number
 *   Uses INADDR_ANY so the server binds to all local interfaces.
 */
void init_server_addr(struct sockaddr_in *addr, int port);

/*
 * Create a TCP listening socket bound to the given port.
 * Returns: listening socket file descriptor on success, -1 on error.
 */
int set_up_server_socket(int port);

/*
 * Accept a pending connection on the listening socket.
 * Returns: new client socket fd on success, -1 on error.
 */
int accept_connection(int server_fd);

/*
 * Create a TCP socket and connect to the given host:port.
 * Returns: connected socket fd on success, -1 on error.
 */
int connect_to_server(const char *host, int port);

#endif
