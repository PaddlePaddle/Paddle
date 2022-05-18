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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "paddle/phi/backends/device_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PD_ExecutionContext PD_ExecutionContext;

typedef struct PD_Tensor PD_Tensor;

typedef C_Status PD_Status;

typedef C_Stream PD_Stream;

typedef C_DataType PD_DataType;

/**
 * internal attribute api
 */

void PD_GetAttrBool(PD_ExecutionContext* ctx,
                    const char* attr_name,
                    bool* attr_val,
                    PD_Status* status);

void PD_GetAttrInt8(PD_ExecutionContext* ctx,
                    const char* attr_name,
                    int8_t* attr_val,
                    PD_Status* status);

void PD_GetAttrInt16(PD_ExecutionContext* ctx,
                     const char* attr_name,
                     int16_t* attr_val,
                     PD_Status* status);

void PD_GetAttrInt32(PD_ExecutionContext* ctx,
                     const char* attr_name,
                     int32_t* attr_val,
                     PD_Status* status);

void PD_GetAttrInt64(PD_ExecutionContext* ctx,
                     const char* attr_name,
                     int64_t* attr_val,
                     PD_Status* status);

void PD_GetAttrUInt8(PD_ExecutionContext* ctx,
                     const char* attr_name,
                     uint8_t* attr_val,
                     PD_Status* status);

void PD_GetAttrUInt16(PD_ExecutionContext* ctx,
                      const char* attr_name,
                      uint16_t* attr_val,
                      PD_Status* status);

void PD_GetAttrUInt32(PD_ExecutionContext* ctx,
                      const char* attr_name,
                      uint32_t* attr_val,
                      PD_Status* status);

void PD_GetAttrUInt64(PD_ExecutionContext* ctx,
                      const char* attr_name,
                      uint64_t* attr_val,
                      PD_Status* status);

// void PD_GetAttrInt32List(PD_ExecutionContext* ctx, const char* attr_name,
//                          int32_t* attr_val, size_t* count, PD_Status*
//                          status);

// void PD_GetAttrInt64List(PD_ExecutionContext* ctx, const char* attr_name,
//                          int64_t* attr_val, size_t* count, PD_Status*
//                          status);

// void PD_GetAttrUInt32List(PD_ExecutionContext* ctx, const char* attr_name,
//                           uint32_t* attr_val, size_t* count, PD_Status*
//                           status);

// void PD_GetAttrUInt64List(PD_ExecutionContext* ctx, const char* attr_name,
//                           uint64_t* attr_val, size_t* count, PD_Status*
//                           status);

/**
 * internal tensor api
 */

size_t PD_NumDims(const PD_Tensor* tensor, PD_Status* status);

size_t PD_Dim(const PD_Tensor* tensor, size_t index, PD_Status* status);

size_t PD_GetOutputNumDims(PD_ExecutionContext* ctx,
                           const char* name,
                           PD_Status* status);

size_t PD_GetOutputDim(PD_ExecutionContext* ctx,
                       const char* name,
                       size_t index,
                       PD_Status* status);

void PD_SetTensorDims(PD_Tensor* tensor,
                      size_t ndims,
                      const size_t* dims,
                      PD_Status* status);

/**
 * public api
 */

PD_Stream PD_GetStream(PD_ExecutionContext* ctx, PD_Status* status);

// size_t PD_NumInputs(PD_ExecutionContext* ctx);

// size_t PD_NumOutputs(PD_ExecutionContext* ctx);

PD_Tensor* PD_GetInput(PD_ExecutionContext* ctx,
                       const char* name,
                       PD_Status* status);

PD_Tensor* PD_GetOutput(PD_ExecutionContext* ctx,
                        const char* name,
                        PD_Status* status);

// void PD_SetOutput(PD_ExecutionContext* ctx, char* name, const PD_Tensor**
// tensor,
//                   PD_Status* status);

// size_t PD_GetAttrSize(PD_ExecutionContext* ctx);

// bool PD_HasAttr(PD_ExecutionContext* ctx, const char* attr_name,
//                 PD_Status* status);

PD_DataType PD_ExpectedOutputDataType(PD_ExecutionContext* ctx,
                                      const char* name,
                                      PD_Status* status);

// void PD_GetOutputShape(PD_ExecutionContext* ctx, const char* name, size_t*
// dims,
//                        size_t* ndims, PD_Status* status);

// PD_Tensor* PD_AllocateOutput(PD_ExecutionContext* ctx, const char* name,
//                              PD_DataType dtype, size_t* dims, size_t ndims,
//                              PD_Status* status);

void PD_AllocateTensor(PD_ExecutionContext* ctx,
                       PD_Tensor* tensor,
                       bool on_host,
                       PD_Status* status);

// PD_Tensor* PD_AllocateTemp(PD_ExecutionContext* ctx, PD_DataType dtype,
//                            size_t* dims, size_t ndims, bool on_host,
//                            PD_Status* status);

// PD_Tensor* PD_NewTensor(PD_DataType dtype, size_t* dims, size_t ndims,
//                         void* data, size_t len, PD_Status* status);

// void PD_DeleteTensor(PD_Tensor* tensor, PD_Status* status);

PD_DataType PD_TensorType(const PD_Tensor* tensor, PD_Status* status);

size_t PD_TensorByteSize(const PD_Tensor* tensor, PD_Status* status);

void* PD_TensorData(const PD_Tensor* tensor, PD_Status* status);

size_t PD_TensorElementCount(const PD_Tensor* tensor, PD_Status* status);

void PD_SetTensorType(PD_Tensor* tensor, PD_DataType dtype, PD_Status* status);

// PD_GetOpKernelInfoMap

void PD_RegisterKernel(const char* kernel,
                       const char* backend,
                       PD_DataType dtype,
                       void (*fn)(PD_ExecutionContext*));

#ifdef __cplusplus
}  // extern "C"
#endif

#include <type_traits>
#include <vector>

#define CPP_TYPE_TO_PD_DTYPE_REGISTER(_) \
  _(bool, PD_DataType::BOOL)             \
  _(float, PD_DataType::FLOAT32)         \
  _(double, PD_DataType::FLOAT64)        \
  _(uint8_t, PD_DataType::UINT8)         \
  _(uint16_t, PD_DataType::UINT16)       \
  _(uint32_t, PD_DataType::UINT32)       \
  _(uint64_t, PD_DataType::UINT64)       \
  _(int8_t, PD_DataType::INT8)           \
  _(int16_t, PD_DataType::INT16)         \
  _(int32_t, PD_DataType::INT32)         \
  _(int64_t, PD_DataType::INT64)

template <typename T>
struct CppTypeToPDType;

#define CPP_TYPE_TO_PD_DTYPE(x, y)                    \
  template <>                                         \
  struct CppTypeToPDType<x> {                         \
    constexpr static PD_DataType Type() { return y; } \
  };

CPP_TYPE_TO_PD_DTYPE_REGISTER(CPP_TYPE_TO_PD_DTYPE)

template <typename T>
T PD_GetAttr(PD_ExecutionContext* ctx,
             const char* attr_name,
             PD_Status* status) {
  T attr_val;
  if (std::is_same<T, int8_t>::value) {
    return PD_GetAttrInt8(
        ctx, attr_name, reinterpret_cast<int8_t*>(&attr_val), status);
  } else if (std::is_same<T, int16_t>::value) {
    return PD_GetAttrInt16(
        ctx, attr_name, reinterpret_cast<int16_t*>(&attr_val), status);
  } else if (std::is_same<T, int32_t>::value) {
    return PD_GetAttrInt32(
        ctx, attr_name, reinterpret_cast<int32_t*>(&attr_val), status);
  } else if (std::is_same<T, int64_t>::value) {
    return PD_GetAttrInt64(
        ctx, attr_name, reinterpret_cast<int64_t*>(&attr_val), status);
  } else if (std::is_same<T, uint8_t>::value) {
    return PD_GetAttrUInt8(
        ctx, attr_name, reinterpret_cast<uint8_t*>(&attr_val), status);
  } else if (std::is_same<T, uint16_t>::value) {
    return PD_GetAttrUInt16(
        ctx, attr_name, reinterpret_cast<uint16_t*>(&attr_val), status);
  } else if (std::is_same<T, uint32_t>::value) {
    return PD_GetAttrUInt32(
        ctx, attr_name, reinterpret_cast<uint32_t*>(&attr_val), status);
  } else if (std::is_same<T, uint64_t>::value) {
    return PD_GetAttrUInt64(
        ctx, attr_name, reinterpret_cast<uint64_t*>(&attr_val), status);
  } else if (std::is_same<T, bool>::value) {
    return PD_GetAttrBool(
        ctx, attr_name, reinterpret_cast<bool*>(&attr_val), status);
  } else {
  }
}

std::vector<size_t> PD_GetTensorShape(PD_Tensor* tensor, PD_Status* status) {
  size_t ndims = PD_NumDims(tensor, status);
  if (ndims > 0) {
    std::vector<size_t> shape(ndims);
    for (size_t i = 0; i < ndims; ++i) {
      shape[i] = PD_Dim(tensor, i, status);
    }
    return shape;
  }
  return std::vector<size_t>();
}

std::vector<size_t> PD_GetOutputShape(PD_ExecutionContext* ctx,
                                      const char* name,
                                      PD_Status* status) {
  size_t ndims = PD_GetOutputNumDims(ctx, name, status);
  if (ndims > 0) {
    std::vector<size_t> shape(ndims);
    for (size_t i = 0; i < ndims; ++i) {
      shape[i] = PD_GetOutputDim(ctx, name, i, status);
    }
    return shape;
  }
  return std::vector<size_t>();
}

void PD_SetTensorShape(PD_Tensor* tensor,
                       const std::vector<size_t>& shape,
                       PD_Status* status) {
  PD_SetTensorDims(tensor, shape.size(), shape.data(), status);
}

class CustomOpKernelBuilder {
 public:
  using KernelFn = void (*)(PD_ExecutionContext*);

  template <typename KernelType>
  CustomOpKernelBuilder& SetKernelFn(KernelFn fn) {
    auto kernel_type = CppTypeToPDType<KernelType>::Type();
    kernels.emplace_back(kernel_type, fn);
    return *this;
  }
  std::vector<std::pair<PD_DataType, KernelFn>> kernels;
};

#include <iostream>
#define PD_BUILD_KERNEL(name, backend)                                   \
  class __custom_op_kernel_registrar_##name##_##backend##_class__ {      \
   public:                                                               \
    __custom_op_kernel_registrar_##name##_##backend##_class__(           \
        CustomOpKernelBuilder& builder) {                                \
      for (auto& kernel : builder.kernels) {                             \
        PD_RegisterKernel(#name, #backend, kernel.first, kernel.second); \
      }                                                                  \
    }                                                                    \
    void Touch() {}                                                      \
  };                                                                     \
  static __custom_op_kernel_registrar_##name##_##backend##_class__       \
      __custom_op_kernel_registrar_##name##_##backend##_instance__ =     \
          CustomOpKernelBuilder()

#endif
