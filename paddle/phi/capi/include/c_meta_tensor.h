// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

typedef struct PD_MetaTensor PD_MetaTensor;

PD_DataType PD_MetaTensorGetPDDataType(const PD_MetaTensor *tensor,
                                       PD_Status *status);

PD_DataLayout PD_MetaTensorGetDataLayout(const PD_MetaTensor *tensor,
                                         PD_Status *status);

int64_t PD_MetaTensorGetElementCount(const PD_MetaTensor *tensor,
                                     PD_Status *status);

int64_t PD_MetaTensorGetNumDims(const PD_MetaTensor *tensor, PD_Status *status);

int64_t PD_MetaTensorGetDim(const PD_MetaTensor *tensor,
                            size_t index,
                            PD_Status *status);

int64_t PD_MetaTensorGetNumStrides(const PD_MetaTensor *tensor,
                                   PD_Status *status);

int64_t PD_MetaTensorGetStride(const PD_MetaTensor *tensor,
                               size_t index,
                               PD_Status *status);

bool PD_MetaTensorIsValid(const PD_MetaTensor *tensor, PD_Status *status);

void PD_MetaTensorSetDims(PD_MetaTensor *tensor,
                          int64_t ndims,
                          const int64_t *dims,
                          PD_Status *status);

void PD_MetaTensorSetStrides(PD_MetaTensor *tensor,
                             int64_t nstrides,
                             const int64_t *strides,
                             PD_Status *status);

void PD_MetaTensorSetDataType(PD_MetaTensor *tensor,
                              PD_DataType dtype,
                              PD_Status *status);

void PD_MetaTensorSetDataLayout(PD_MetaTensor *tensor,
                                PD_DataLayout layout,
                                PD_Status *status);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
