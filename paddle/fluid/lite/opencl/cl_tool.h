/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/lite/opencl/cl_include.h"

namespace paddle {
namespace lite {

const char* opencl_error_to_str(cl_int error);

#define CL_CHECK_ERRORS(ERR)                                         \
  if (ERR != CL_SUCCESS) {                                           \
    printf(                                                          \
        "OpenCL error with code %s happened in file %s at line %d. " \
        "Exiting.\n",                                                \
        opencl_error_to_str(ERR), __FILE__, __LINE__);               \
  }
}  // namespace lite
}  // namespace paddle
