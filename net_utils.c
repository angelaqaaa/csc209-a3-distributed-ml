#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>

#include "net_utils.h"

void init_server_addr(struct sockaddr_in *addr, int port) {
    memset(addr, 0, sizeof(*addr));
    addr->sin_family = AF_INET;
    addr->sin_addr.s_addr = htonl(INADDR_ANY);
    addr->sin_port = htons(port);
}


int set_up_server_socket(int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    int opt = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(sockfd);
        return -1;
    }

    struct sockaddr_in serv_addr;
    init_server_addr(&serv_addr, port);

    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("bind");
        close(sockfd);
        return -1;
    }

    if (listen(sockfd, 16) < 0) {
        perror("listen");
        close(sockfd);
        return -1;
    }

    return sockfd;
}


int accept_connection(int server_fd) {
    struct sockaddr_in cli_addr;
    socklen_t cli_len = sizeof(cli_addr);
    int client_fd = accept(server_fd, (struct sockaddr *)&cli_addr, &cli_len);
    if (client_fd < 0) {
        perror("accept");
        return -1;
    }
    return client_fd;
}


int connect_to_server(const char *host, int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host, &serv_addr.sin_addr) <= 0) {
        /* Not a numeric IP -- try DNS lookup as a fallback */
        struct hostent *h = gethostbyname(host);
        if (h && h->h_addr_list && h->h_addr_list[0]) {
            memcpy(&serv_addr.sin_addr.s_addr, h->h_addr_list[0],
                   h->h_length);
        } else if (strcmp(host, "localhost") == 0) {
            serv_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        } else {
            fprintf(stderr,
                    "invalid host/address or DNS lookup failed: %s\n",
                    host);
            close(sockfd);
            return -1;
        }
    }

    if (connect(sockfd, (struct sockaddr *)&serv_addr,
                sizeof(serv_addr)) < 0) {
        perror("connect");
        close(sockfd);
        return -1;
    }

    return sockfd;
}
