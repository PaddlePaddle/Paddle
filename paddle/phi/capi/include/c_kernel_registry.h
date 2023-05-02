// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if !defined(_WIN32)

#include <vector>

#include "paddle/phi/capi/include/c_data_type.h"
#include "paddle/phi/capi/include/c_kernel_context.h"
#include "paddle/phi/capi/include/c_kernel_factory.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  PD_ARG_TYPE_CONTEXT = 0,
  PD_ARG_TYPE_TENSOR,
  PD_ARG_TYPE_BOOL,
  PD_ARG_TYPE_BFLOAT16,
  PD_ARG_TYPE_FLOAT16,
  PD_ARG_TYPE_FLOAT32,
  PD_ARG_TYPE_FLOAT64,
  PD_ARG_TYPE_INT32,
  PD_ARG_TYPE_INT64,
  PD_ARG_TYPE_STRING,
  PD_ARG_TYPE_SCALAR,
  PD_ARG_TYPE_INT_ARRAY,
  PD_ARG_TYPE_DATA_TYPE,
  PD_ARG_TYPE_DATA_LAYOUT,
  PD_ARG_TYPE_PLACE,
  PD_ARG_TYPE_LIST_BOOL,
  PD_ARG_TYPE_LIST_INT32,
  PD_ARG_TYPE_LIST_INT64,
  PD_ARG_TYPE_LIST_BFLOAT16,
  PD_ARG_TYPE_LIST_FLOAT16,
  PD_ARG_TYPE_LIST_FLOAT32,
  PD_ARG_TYPE_LIST_FLOAT64,
  PD_ARG_TYPE_LIST_STRING,
  PD_ARG_TYPE_LIST_SCALAR,
  PD_ARG_TYPE_OPTIONAL_TENSOR,
  PD_ARG_TYPE_LIST_TENSOR,
  PD_ARG_TYPE_OPTIONAL_MULTI_TENSOR,
} PD_KernelArgumentType;

void PD_RegisterPhiKernel(const char *kernel_name_cstr,
                          const char *backend_cstr,
                          PD_DataType pd_dtype,
                          PD_DataLayout pd_layout,
                          size_t in_nargs,
                          PD_KernelArgumentType *in_args_type,
                          size_t attr_nargs,
                          PD_KernelArgumentType *attr_args_type,
                          size_t out_nargs,
                          PD_KernelArgumentType *out_args_type,
                          void (*args_def_fn)(const PD_KernelKey *,
                                              PD_Kernel *),
                          void (*fn)(PD_KernelContext *),
                          void *variadic_kernel_fn);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
