/* Partner 2 drafts skeleton; both partners add their functions */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/select.h>
#include "protocol.h"
#include "net_utils.h"
#include "io_utils.h"
#include "model.h"

/* ========== CONNECTION MANAGEMENT (Partner 2) ========== */
/* TODO: add_worker() */
/* TODO: handle_register() */
/* TODO: broadcast_weights() */

/* ========== TRAINING LOGIC (Partner 1) ========== */
/* TODO: handle_gradient() */
/* TODO: all_gradients_received() */
/* TODO: aggregate_and_update() */
/* TODO: check_termination() */
/* TODO: broadcast_done() */

/* ========== SHARED (skeleton) ========== */
/* TODO: dispatch_message() */
/* TODO: handle_disconnect() */
/* TODO: main() — select() loop following simpleselect.c pattern */
