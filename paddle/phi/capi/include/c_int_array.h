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

#include "paddle/phi/capi/include/c_data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PD_IntArray PD_IntArray;

PD_List PD_IntArrayGetDataPointer(PD_IntArray *int_array);

size_t PD_IntArrayGetElementCount(PD_IntArray *int_array);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
