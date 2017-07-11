#pragma once

#include "dll_exports.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* paddle_variable_handle;

paddle_error paddle_destroy_variable(paddle_variable_handle);

#ifdef __cplusplus
};
#endif
