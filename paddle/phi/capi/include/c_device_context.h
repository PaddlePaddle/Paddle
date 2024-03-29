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
#include "paddle/phi/capi/include/c_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PD_DeviceContext PD_DeviceContext;

typedef C_Stream PD_Stream;

PD_Stream PD_DeviceContextGetStream(const PD_DeviceContext *ctx,
                                    PD_Status *status);

void *PD_DeviceContextAllocateTensor(const PD_DeviceContext *ctx,
                                     PD_Tensor *tensor,
                                     size_t size,
                                     PD_DataType dtype,
                                     PD_Status *status);

void PD_DeviceContextSetSeed(const PD_DeviceContext *ctx,
                             uint64_t seed,
                             PD_Status *status);

uint64_t PD_DeviceContextGetSeed(const PD_DeviceContext *ctx,
                                 PD_Status *status);

uint64_t PD_DeviceContextGetRandom(const PD_DeviceContext *ctx,
                                   PD_Status *status);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
