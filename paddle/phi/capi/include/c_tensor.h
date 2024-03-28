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

typedef struct PD_Tensor PD_Tensor;

PD_DataType PD_TensorGetPDDataType(const PD_Tensor *tensor, PD_Status *status);

PD_DataLayout PD_TensorGetDataLayout(const PD_Tensor *tensor,
                                     PD_Status *status);

int64_t PD_TensorGetByteSize(const PD_Tensor *tensor, PD_Status *status);

void *PD_TensorGetDataPointer(const PD_Tensor *tensor, PD_Status *status);

int64_t PD_TensorGetElementCount(const PD_Tensor *tensor, PD_Status *status);

int64_t PD_TensorGetNumDims(const PD_Tensor *tensor, PD_Status *status);

int64_t PD_TensorGetDim(const PD_Tensor *tensor,
                        size_t index,
                        PD_Status *status);

int64_t PD_TensorGetNumStrides(const PD_Tensor *tensor, PD_Status *status);

int64_t PD_TensorGetStride(const PD_Tensor *tensor,
                           size_t index,
                           PD_Status *status);

void PD_TensorGetLoD(const PD_Tensor *tensor,
                     PD_List *data,
                     PD_List *offset,
                     PD_Status *status);

bool PD_TensorIsInitialized(const PD_Tensor *tensor, PD_Status *status);

bool PD_TensorIsValid(const PD_Tensor *tensor, PD_Status *status);

void *PD_TensorGetHolder(const PD_Tensor *tensor, PD_Status *status);

size_t PD_TensorGetOffset(const PD_Tensor *tensor, PD_Status *status);

void PD_TensorSetDims(PD_Tensor *tensor,
                      int64_t ndims,
                      const int64_t *dims,
                      PD_Status *status);

void PD_TensorSetOffset(PD_Tensor *tensor,
                        const int64_t offset,
                        PD_Status *status);

void PD_TensorSetStrides(PD_Tensor *tensor,
                         int64_t nstrides,
                         const int64_t *strides,
                         PD_Status *status);

void PD_TensorSetDataType(PD_Tensor *tensor,
                          PD_DataType dtype,
                          PD_Status *status);

void PD_TensorSetDataLayout(PD_Tensor *tensor,
                            PD_DataLayout layout,
                            PD_Status *status);

void PD_TensorResetLoD(PD_Tensor *tensor,
                       PD_List data,
                       PD_List offset,
                       PD_Status *status);

PD_Tensor *PD_NewTensor();

void PD_DeleteTensor(PD_Tensor *tensor);

void PD_TensorShareDataWith(PD_Tensor *dst,
                            const PD_Tensor *src,
                            PD_Status *status);

void PD_TensorShareLoDWith(PD_Tensor *dst,
                           const PD_Tensor *src,
                           PD_Status *status);

PD_Tensor *PD_OptionalTensorGetPointer(PD_Tensor *tensor);

PD_List PD_TensorVectorToList(PD_Tensor *tensor);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
