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

#if !defined(_WIN32) && !defined(__APPLE__)

#include <cstdint>

#include "paddle/phi/backends/device_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PD_DeviceContext PD_DeviceContext;

typedef struct PD_ExecutionContext PD_ExecutionContext;

typedef struct PD_Tensor PD_Tensor;

typedef struct PD_Scalar PD_Scalar;

typedef struct PD_IntArray PD_IntArray;

typedef C_Status PD_Status;

typedef C_Stream PD_Stream;

typedef C_DataType PD_DataType;

typedef C_DataLayout PD_DataLayout;

typedef struct PD_KernelKey PD_KernelKey;

typedef struct PD_Kernel PD_Kernel;

typedef struct PD_Place PD_Place;

typedef struct PD_KernelArgsDef PD_KernelArgsDef;

typedef struct PD_TensorArgDef PD_TensorArgDef;

typedef struct {
  size_t size;
  void *data;
} PD_List;

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
} PD_ArgumentType;

void PD_RegisterPhiKernel(const char *kernel_name_cstr,
                          const char *backend_cstr,
                          PD_DataType pd_dtype,
                          PD_DataLayout pd_layout,
                          size_t in_nargs,
                          PD_ArgumentType *in_args_type,
                          size_t attr_nargs,
                          PD_ArgumentType *attr_args_type,
                          size_t out_nargs,
                          PD_ArgumentType *out_args_type,
                          void (*args_def_fn)(const PD_KernelKey *,
                                              PD_Kernel *),
                          void (*fn)(PD_ExecutionContext *),
                          void *variadic_kernel_fn);

PD_DeviceContext *PD_OriginGetDeviceContext(PD_ExecutionContext *ctx);

PD_Tensor *PD_OriginInputAt(PD_ExecutionContext *ctx, size_t index);

PD_Tensor *PD_OriginOptionalInputAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_OriginMultiInputAt(PD_ExecutionContext *ctx, size_t index);

PD_Tensor *PD_OriginOutputAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_OriginMultiOutputAt(PD_ExecutionContext *ctx, size_t index);

/**
 * Attribute
 */

bool PD_BoolAttrAt(PD_ExecutionContext *ctx, size_t index);

int32_t PD_Int32AttrAt(PD_ExecutionContext *ctx, size_t index);

int64_t PD_Int64AttrAt(PD_ExecutionContext *ctx, size_t index);

float PD_FloatAttrAt(PD_ExecutionContext *ctx, size_t index);

double PD_DoubleAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_Scalar *PD_ScalarAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_IntArray *PD_IntArrayAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_DataType PD_DataTypeAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_DataLayout PD_DataLayoutAttrAt(PD_ExecutionContext *ctx, size_t index);

char *PD_StringAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_ListBoolAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_ListInt32AttrAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_ListInt64AttrAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_ListFloatAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_ListDoubleAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_ListStringAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_List PD_ListScalarAttrAt(PD_ExecutionContext *ctx, size_t index);

PD_Place *PD_PlaceAttrAt(PD_ExecutionContext *ctx, size_t index);

/**
 * DeviceContext
 */

PD_Stream PD_DeviceContextGetStream(const PD_DeviceContext *ctx,
                                    PD_Status *status);

void *PD_DeviceContextAllocateTensor(const PD_DeviceContext *ctx,
                                     PD_Tensor *tensor,
                                     size_t size,
                                     PD_DataType dtype,
                                     PD_Status *status);

/**
 * Tensor
 */

PD_DataType PD_TensorGetDataType(const PD_Tensor *tensor, PD_Status *status);

PD_DataLayout PD_TensorGetDataLayout(const PD_Tensor *tensor,
                                     PD_Status *status);

int64_t PD_TensorGetByteSize(const PD_Tensor *tensor, PD_Status *status);

void *PD_TensorGetDataPointer(const PD_Tensor *tensor, PD_Status *status);

int64_t PD_TensorGetElementCount(const PD_Tensor *tensor, PD_Status *status);

int64_t PD_TensorGetNumDims(const PD_Tensor *tensor, PD_Status *status);

int64_t PD_TensorGetDim(const PD_Tensor *tensor,
                        size_t index,
                        PD_Status *status);

void PD_TensorGetLoD(const PD_Tensor *tensor,
                     PD_List *data,
                     PD_List *offset,
                     PD_Status *status);

bool PD_TensorIsInitialized(const PD_Tensor *tensor, PD_Status *status);

bool PD_TensorIsValid(const PD_Tensor *tensor, PD_Status *status);

void *PD_TensorGetHolder(const PD_Tensor *tensor, PD_Status *status);

void PD_TensorSetDims(PD_Tensor *tensor,
                      int64_t ndims,
                      const int64_t *dims,
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

/**
 * Scalar
 */

bool PD_ScalarGetBoolData(PD_Scalar *scalar);

int8_t PD_ScalarGetInt8Data(PD_Scalar *scalar);

int16_t PD_ScalarGetInt16Data(PD_Scalar *scalar);

int32_t PD_ScalarGetInt32Data(PD_Scalar *scalar);

int64_t PD_ScalarGetInt64Data(PD_Scalar *scalar);

uint8_t PD_ScalarGetUInt8Data(PD_Scalar *scalar);

uint16_t PD_ScalarGetUInt16Data(PD_Scalar *scalar);

uint32_t PD_ScalarGetUInt32Data(PD_Scalar *scalar);

uint64_t PD_ScalarGetUInt64Data(PD_Scalar *scalar);

float PD_ScalarGetFloat32Data(PD_Scalar *scalar);

double PD_ScalarGetFloat64Data(PD_Scalar *scalar);

PD_DataType PD_ScalarGetDataType(PD_Scalar *scalar);

/**
 * IntArray
 */

PD_List PD_IntArrayGetDataPointer(PD_IntArray *int_array);

size_t PD_IntArrayGetElementCount(PD_IntArray *int_array);

/**
 * PD_List
 */

void PD_DeleteList(PD_List list);

void PD_DeleteUInt8List(PD_List list);

void PD_DeleteInt64List(PD_List list);

void PD_DeleteInt32List(PD_List list);

void PD_DeleteFloat64List(PD_List list);

void PD_DeleteFloat32List(PD_List list);

/**
 * Place
 */

bool PD_PlaceIsHost(PD_Place *place);

int8_t PD_PlaceGetDeviceId(PD_Place *place);

/**
 * TensorArgDef
 */

void PD_TensorArgDefSetDataLayout(PD_TensorArgDef *def,
                                  PD_DataLayout layout,
                                  PD_Status *status);

void PD_TensorArgDefSetDataType(PD_TensorArgDef *def,
                                PD_DataType dtype,
                                PD_Status *status);

/**
 * KernelArgsDef
 */

PD_List PD_KernelArgsDefGetInputArgDefs(PD_KernelArgsDef *def,
                                        PD_Status *status);

PD_List PD_KernelArgsDefGetOutputArgDefs(PD_KernelArgsDef *def,
                                         PD_Status *status);

/**
 * KernelKey
 */

PD_DataLayout PD_KernelKeyGetLayout(PD_KernelKey *key, PD_Status *status);

PD_DataType PD_KernelKeyGetDataType(PD_KernelKey *key, PD_Status *status);

/**
 * Kernel
 */

PD_KernelArgsDef *PD_KernelGetArgsDef(PD_Kernel *kernel, PD_Status *status);

#ifdef __cplusplus
}  // extern "C"
#endif

#include <vector>

template <typename T>
static inline PD_List PDListFromVector(std::vector<T> *vec) {
  PD_List list;
  list.data = reinterpret_cast<void *>(vec->data());
  list.size = vec->size();
  return list;
}

template <typename T>
static inline std::vector<T> PDListToVector(PD_List list) {
  return std::vector<T>(static_cast<T *>(list.data),
                        static_cast<T *>(list.data) + list.size);
}

#endif
