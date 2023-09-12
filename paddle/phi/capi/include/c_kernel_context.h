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
#include "paddle/phi/capi/include/c_device_context.h"
#include "paddle/phi/capi/include/c_int_array.h"
#include "paddle/phi/capi/include/c_place.h"
#include "paddle/phi/capi/include/c_scalar.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PD_KernelContext PD_KernelContext;

/**
 * KernelContext
 */

PD_DeviceContext *PD_KernelContextGetDeviceContext(PD_KernelContext *ctx);

PD_Tensor *PD_KernelContextInputAt(PD_KernelContext *ctx, size_t index);

// PD_Tensor *PD_KernelContextOptionalInputAt(PD_KernelContext *ctx, size_t
// index);

PD_List PD_KernelContextMultiInputAt(PD_KernelContext *ctx, size_t index);

PD_Tensor *PD_KernelContextOutputAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextMultiOutputAt(PD_KernelContext *ctx, size_t index);

/**
 * Attribute
 */

bool PD_KernelContextBoolAttrAt(PD_KernelContext *ctx, size_t index);

int32_t PD_KernelContextInt32AttrAt(PD_KernelContext *ctx, size_t index);

int64_t PD_KernelContextInt64AttrAt(PD_KernelContext *ctx, size_t index);

float PD_KernelContextFloatAttrAt(PD_KernelContext *ctx, size_t index);

double PD_KernelContextDoubleAttrAt(PD_KernelContext *ctx, size_t index);

PD_Scalar *PD_KernelContextScalarAttrAt(PD_KernelContext *ctx, size_t index);

PD_IntArray *PD_KernelContextIntArrayAttrAt(PD_KernelContext *ctx,
                                            size_t index);

PD_DataType PD_KernelContextDataTypeAttrAt(PD_KernelContext *ctx, size_t index);

PD_DataLayout PD_KernelContextDataLayoutAttrAt(PD_KernelContext *ctx,
                                               size_t index);

char *PD_KernelContextStringAttrAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextListBoolAttrAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextListInt32AttrAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextListInt64AttrAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextListFloatAttrAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextListDoubleAttrAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextListStringAttrAt(PD_KernelContext *ctx, size_t index);

PD_List PD_KernelContextListScalarAttrAt(PD_KernelContext *ctx, size_t index);

PD_Place *PD_KernelContextPlaceAttrAt(PD_KernelContext *ctx, size_t index);

const char *PD_StringAttr(void *attr);

PD_DataType PD_DatatTypeAttr(void *attr);

PD_DataLayout PD_DatatLayoutAttr(void *attr);

PD_List PD_ListInt32Attr(void *attr);

PD_List PD_ListInt64Attr(void *attr);

PD_List PD_ListFloatAttr(void *attr);

PD_List PD_ListDoubleAttr(void *attr);

PD_List PD_ListScalarAttr(void *attr);

PD_List PD_ListStringAttr(void *attr);

PD_List PD_ListBoolAttr(void *attr);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
