#ifndef __PADDLE_CAPI_MAIN_H__
#define __PADDLE_CAPI_MAIN_H__
#include "config.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize Paddle.
 */
PD_API paddle_error paddle_init(int argc, char** argv);

#ifdef __cplusplus
}
#endif

#endif
