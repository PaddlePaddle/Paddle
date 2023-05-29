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
#include <cstdint>

#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef C_Status PD_Status;

typedef C_DataType PD_DataType;

typedef C_DataLayout PD_DataLayout;

typedef struct {
  size_t size;
  void *data;
} PD_List;

void PD_DeletePointerList(PD_List list);

void PD_DeleteUInt8List(PD_List list);

void PD_DeleteInt64List(PD_List list);

void PD_DeleteInt32List(PD_List list);

void PD_DeleteFloat64List(PD_List list);

void PD_DeleteFloat32List(PD_List list);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
