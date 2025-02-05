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
#include "paddle/phi/capi/include/c_int_array.h"
#include "paddle/phi/capi/include/c_meta_tensor.h"
#include "paddle/phi/capi/include/c_place.h"
#include "paddle/phi/capi/include/c_scalar.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PD_InferMetaContext PD_InferMetaContext;

PD_MetaTensor *PD_InferMetaContextInputAt(PD_InferMetaContext *ctx,
                                          size_t index);

PD_List PD_InferMetaContextMultiInputAt(PD_InferMetaContext *ctx, size_t index);

PD_MetaTensor *PD_InferMetaContextOutputAt(PD_InferMetaContext *ctx,
                                           size_t index);

PD_List PD_InferMetaContextMultiOutputAt(PD_InferMetaContext *ctx,
                                         size_t index);

bool PD_InferMetaContextBoolAttrAt(PD_InferMetaContext *ctx, size_t index);

int32_t PD_InferMetaContextInt32AttrAt(PD_InferMetaContext *ctx, size_t index);

int64_t PD_InferMetaContextInt64AttrAt(PD_InferMetaContext *ctx, size_t index);

float PD_InferMetaContextFloatAttrAt(PD_InferMetaContext *ctx, size_t index);

double PD_InferMetaContextDoubleAttrAt(PD_InferMetaContext *ctx, size_t index);

PD_Scalar *PD_InferMetaContextScalarAttrAt(PD_InferMetaContext *ctx,
                                           size_t index);

PD_IntArray *PD_InferMetaContextIntArrayAttrAt(PD_InferMetaContext *ctx,
                                               size_t index);

PD_DataType PD_InferMetaContextDataTypeAttrAt(PD_InferMetaContext *ctx,
                                              size_t index);

PD_DataLayout PD_InferMetaContextDataLayoutAttrAt(PD_InferMetaContext *ctx,
                                                  size_t index);

char *PD_InferMetaContextStringAttrAt(PD_InferMetaContext *ctx, size_t index);

PD_List PD_InferMetaContextListBoolAttrAt(PD_InferMetaContext *ctx,
                                          size_t index);

PD_List PD_InferMetaContextListInt32AttrAt(PD_InferMetaContext *ctx,
                                           size_t index);

PD_List PD_InferMetaContextListInt64AttrAt(PD_InferMetaContext *ctx,
                                           size_t index);

PD_List PD_InferMetaContextListFloatAttrAt(PD_InferMetaContext *ctx,
                                           size_t index);

PD_List PD_InferMetaContextListDoubleAttrAt(PD_InferMetaContext *ctx,
                                            size_t index);

PD_List PD_InferMetaContextListStringAttrAt(PD_InferMetaContext *ctx,
                                            size_t index);

PD_List PD_InferMetaContextListScalarAttrAt(PD_InferMetaContext *ctx,
                                            size_t index);

PD_Place *PD_InferMetaContextPlaceAttrAt(PD_InferMetaContext *ctx,
                                         size_t index);

PD_DataType PD_InferMetaContextDataTypeAttrAt(PD_InferMetaContext *ctx,
                                              size_t index);

PD_DataLayout PD_InferMetaContextDataLayoutAttrAt(PD_InferMetaContext *ctx,
                                                  size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
