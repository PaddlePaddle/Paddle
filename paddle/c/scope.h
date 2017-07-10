/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "dll_exports.h"
#include "error.h"
#include "variable.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* paddle_scope_handle;

paddle_scope_handle PADDLE_API paddle_new_scope();

paddle_error PADDLE_API paddle_new_scope_with_parent(
    paddle_scope_handle parent, paddle_scope_handle* new_scope);

paddle_error PADDLE_API paddle_destroy_scope(paddle_scope_handle);

paddle_error PADDLE_API paddle_scope_get_var(paddle_scope_handle scope,
                                             const char* name,
                                             paddle_variable_handle* var);

paddle_error PADDLE_API paddle_scope_create_var(paddle_scope_handle scope,
                                                const char* name,
                                                paddle_variable_handle* var);

#ifdef __cplusplus
};
#endif
