#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include "io_utils.h"
#include "protocol.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
        printf("  PASS: %s\n", msg); \
    } else { \
        printf("  FAIL: %s\n", msg); \
    } \
} while (0)

/* ========== read_all tests ========== */

static void test_read_all_basic(void) {
    int pipefd[2];
    char buf[16];

    printf("test_read_all_basic\n");
    pipe(pipefd);

    write(pipefd[1], "hello", 5);
    close(pipefd[1]);

    int ret = read_all(pipefd[0], buf, 5);
    TEST_ASSERT(ret == 0, "read_all returns 0 on success");
    TEST_ASSERT(memcmp(buf, "hello", 5) == 0, "read_all reads correct data");
    close(pipefd[0]);
}

static void test_read_all_eof(void) {
    int pipefd[2];
    char buf[16];

    printf("test_read_all_eof\n");
    pipe(pipefd);

    /* Write 3 bytes but request 5 — should get EOF mid-read */
    write(pipefd[1], "abc", 3);
    close(pipefd[1]);

    int ret = read_all(pipefd[0], buf, 5);
    TEST_ASSERT(ret == -1, "read_all returns -1 on premature EOF");
    close(pipefd[0]);
}

static void test_read_all_closed_fd(void) {
    char buf[16];

    printf("test_read_all_closed_fd\n");
    int ret = read_all(-1, buf, 5);
    TEST_ASSERT(ret == -1, "read_all returns -1 on invalid fd");
}

static void test_read_all_zero_bytes(void) {
    int pipefd[2];
    char buf[1];

    printf("test_read_all_zero_bytes\n");
    pipe(pipefd);
    close(pipefd[1]);

    /* Reading 0 bytes should succeed immediately (loop body never runs) */
    int ret = read_all(pipefd[0], buf, 0);
    TEST_ASSERT(ret == 0, "read_all with 0 bytes returns 0");
    close(pipefd[0]);
}

/* ========== write_all tests ========== */

static void test_write_all_basic(void) {
    int pipefd[2];
    char buf[16];

    printf("test_write_all_basic\n");
    pipe(pipefd);

    int ret = write_all(pipefd[1], "world", 5);
    TEST_ASSERT(ret == 0, "write_all returns 0 on success");

    close(pipefd[1]);
    int n = read(pipefd[0], buf, 5);
    TEST_ASSERT(n == 5 && memcmp(buf, "world", 5) == 0,
                "write_all data is readable");
    close(pipefd[0]);
}

static void test_write_all_closed_reader(void) {
    int pipefd[2];

    printf("test_write_all_closed_reader\n");
    pipe(pipefd);
    close(pipefd[0]);  /* close read end */

    signal(SIGPIPE, SIG_IGN);
    int ret = write_all(pipefd[1], "data", 4);
    TEST_ASSERT(ret == -1, "write_all returns -1 when reader closed");
    close(pipefd[1]);
}

static void test_write_all_zero_bytes(void) {
    int pipefd[2];

    printf("test_write_all_zero_bytes\n");
    pipe(pipefd);

    int ret = write_all(pipefd[1], "x", 0);
    TEST_ASSERT(ret == 0, "write_all with 0 bytes returns 0");

    close(pipefd[1]);
    close(pipefd[0]);
}

/* ========== accumulate_read tests ========== */

static void test_accumulate_read_complete_msg(void) {
    int sv[2];
    struct worker_info w;

    printf("test_accumulate_read_complete_msg\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    memset(&w, 0, sizeof(w));
    w.fd = sv[0];
    w.recv_len = 0;

    /* Build a complete message: type=1, payload_size=4, payload="ABCD" */
    char msg[9];
    msg[0] = MSG_REGISTER;
    uint32_t plen = htonl(4);
    memcpy(msg + 1, &plen, 4);
    memcpy(msg + 5, "ABCD", 4);
    write(sv[1], msg, 9);

    int ret = accumulate_read(&w);
    TEST_ASSERT(ret == 1, "accumulate_read returns 1 for complete message");
    TEST_ASSERT(w.recv_len == 9, "recv_len equals total message size");

    close(sv[0]);
    close(sv[1]);
}

static void test_accumulate_read_partial_header(void) {
    int sv[2];
    struct worker_info w;

    printf("test_accumulate_read_partial_header\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    memset(&w, 0, sizeof(w));
    w.fd = sv[0];
    w.recv_len = 0;

    /* Send only 3 bytes (less than HEADER_SIZE=5) */
    write(sv[1], "abc", 3);

    int ret = accumulate_read(&w);
    TEST_ASSERT(ret == 0, "accumulate_read returns 0 for partial header");
    TEST_ASSERT(w.recv_len == 3, "recv_len updated to 3");

    close(sv[0]);
    close(sv[1]);
}

static void test_accumulate_read_partial_payload(void) {
    int sv[2];
    struct worker_info w;

    printf("test_accumulate_read_partial_payload\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    memset(&w, 0, sizeof(w));
    w.fd = sv[0];
    w.recv_len = 0;

    /* Send header (says 8 bytes of payload) but only 2 bytes of payload */
    char msg[7];
    msg[0] = MSG_GRADIENT;
    uint32_t plen = htonl(8);
    memcpy(msg + 1, &plen, 4);
    msg[5] = 'X';
    msg[6] = 'Y';
    write(sv[1], msg, 7);

    int ret = accumulate_read(&w);
    TEST_ASSERT(ret == 0, "accumulate_read returns 0 for partial payload");
    TEST_ASSERT(w.recv_len == 7, "recv_len updated to 7");

    close(sv[0]);
    close(sv[1]);
}

static void test_accumulate_read_disconnect(void) {
    int sv[2];
    struct worker_info w;

    printf("test_accumulate_read_disconnect\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    memset(&w, 0, sizeof(w));
    w.fd = sv[0];
    w.recv_len = 0;

    close(sv[1]);  /* close writer = EOF */

    int ret = accumulate_read(&w);
    TEST_ASSERT(ret == -1, "accumulate_read returns -1 on disconnect");

    close(sv[0]);
}

static void test_accumulate_read_multi_call(void) {
    int sv[2];
    struct worker_info w;

    printf("test_accumulate_read_multi_call\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    memset(&w, 0, sizeof(w));
    w.fd = sv[0];
    w.recv_len = 0;

    /* Send header only first */
    char hdr[5];
    hdr[0] = MSG_DONE;
    uint32_t plen = htonl(4);
    memcpy(hdr + 1, &plen, 4);
    write(sv[1], hdr, 5);

    int ret1 = accumulate_read(&w);
    TEST_ASSERT(ret1 == 0, "first call returns 0 (header only, no payload)");

    /* Now send the 4-byte payload */
    write(sv[1], "DONE", 4);
    int ret2 = accumulate_read(&w);
    TEST_ASSERT(ret2 == 1, "second call returns 1 (message complete)");
    TEST_ASSERT(w.recv_len == 9, "total recv_len is 9");

    close(sv[0]);
    close(sv[1]);
}

/* ========== send_header / read_header tests ========== */

static void test_header_roundtrip(void) {
    int sv[2];
    uint8_t rbuf[HEADER_SIZE];

    printf("test_header_roundtrip\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    int ret = send_header(sv[1], MSG_GRADIENT, 256);
    TEST_ASSERT(ret == 0, "send_header returns 0 on success");

    close(sv[1]);
    int n = read(sv[0], rbuf, HEADER_SIZE);
    TEST_ASSERT(n == HEADER_SIZE, "read back HEADER_SIZE bytes");

    uint8_t type;
    uint32_t payload_len;
    read_header(rbuf, &type, &payload_len);
    TEST_ASSERT(type == MSG_GRADIENT, "round-trip type is MSG_GRADIENT");
    TEST_ASSERT(payload_len == 256, "round-trip payload_len is 256");

    close(sv[0]);
}

static void test_header_all_types(void) {
    uint8_t types[] = {MSG_REGISTER, MSG_WEIGHTS, MSG_GRADIENT, MSG_DONE};
    const char *names[] = {"REGISTER", "WEIGHTS", "GRADIENT", "DONE"};
    int i;

    printf("test_header_all_types\n");
    for (i = 0; i < 4; i++) {
        int sv[2];
        socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

        send_header(sv[1], types[i], 100);
        close(sv[1]);

        uint8_t rbuf[HEADER_SIZE];
        read(sv[0], rbuf, HEADER_SIZE);

        uint8_t got_type;
        uint32_t got_len;
        read_header(rbuf, &got_type, &got_len);

        char msg[64];
        snprintf(msg, sizeof(msg), "round-trip type %s", names[i]);
        TEST_ASSERT(got_type == types[i], msg);
        TEST_ASSERT(got_len == 100, "round-trip payload len for type");

        close(sv[0]);
    }
}

static void test_header_zero_payload(void) {
    printf("test_header_zero_payload\n");

    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    send_header(sv[1], MSG_DONE, 0);
    close(sv[1]);

    uint8_t rbuf[HEADER_SIZE];
    read(sv[0], rbuf, HEADER_SIZE);

    uint8_t type;
    uint32_t payload_len;
    read_header(rbuf, &type, &payload_len);
    TEST_ASSERT(payload_len == 0, "zero payload_len round-trips correctly");
    close(sv[0]);
}

static void test_header_large_payload(void) {
    printf("test_header_large_payload\n");

    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    uint32_t big = 0x01020304;
    send_header(sv[1], MSG_WEIGHTS, big);
    close(sv[1]);

    uint8_t rbuf[HEADER_SIZE];
    read(sv[0], rbuf, HEADER_SIZE);

    uint8_t type;
    uint32_t payload_len;
    read_header(rbuf, &type, &payload_len);
    TEST_ASSERT(payload_len == big, "large payload_len round-trips correctly");
    close(sv[0]);
}

static void test_send_header_closed_fd(void) {
    printf("test_send_header_closed_fd\n");
    signal(SIGPIPE, SIG_IGN);
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    close(sv[0]);
    int ret = send_header(sv[1], MSG_DONE, 0);
    TEST_ASSERT(ret == -1, "send_header returns -1 on closed peer");
    close(sv[1]);
}

static void test_read_header_network_order(void) {
    printf("test_read_header_network_order\n");

    /* Manually construct header with known network byte order */
    uint8_t buf[HEADER_SIZE];
    buf[0] = MSG_REGISTER;
    uint32_t net_val = htonl(42);
    memcpy(buf + 1, &net_val, 4);

    uint8_t type;
    uint32_t payload_len;
    read_header(buf, &type, &payload_len);
    TEST_ASSERT(type == MSG_REGISTER, "manual header type correct");
    TEST_ASSERT(payload_len == 42, "manual header payload_len correct (42)");
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);

    printf("=== io_utils tests ===\n\n");

    test_read_all_basic();
    test_read_all_eof();
    test_read_all_closed_fd();
    test_read_all_zero_bytes();

    test_write_all_basic();
    test_write_all_closed_reader();
    test_write_all_zero_bytes();

    test_accumulate_read_complete_msg();
    test_accumulate_read_partial_header();
    test_accumulate_read_partial_payload();
    test_accumulate_read_disconnect();
    test_accumulate_read_multi_call();

    test_header_roundtrip();
    test_header_all_types();
    test_header_zero_payload();
    test_header_large_payload();
    test_send_header_closed_fd();
    test_read_header_network_order();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
