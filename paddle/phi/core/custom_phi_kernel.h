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
#include <memory>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/custom_phi_kernel_c_api.h"
#include "paddle/utils/optional.h"

#define PD_CHECK(status)

inline std::vector<size_t> PD_GetTensorDims(PD_Tensor* tensor,
                                            PD_Status* status) {
  size_t ndims = PD_GetTensorNumDims(tensor, status);
  if (ndims > 0) {
    std::vector<size_t> shape(ndims);
    for (size_t i = 0; i < ndims; ++i) {
      shape[i] = PD_GetTensorDim(tensor, i, status);
    }
    return shape;
  }
  return std::vector<size_t>();
}

namespace phi {

// utils

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

template <PD_DataType T>
struct PDTypeToCppType;

#define PD_DTYPE_TO_CPP_TYPE(x, y) \
  template <>                      \
  struct PDTypeToCppType<y> {      \
    using type = x;                \
  };

CPP_TYPE_TO_PD_DTYPE_REGISTER(CPP_TYPE_TO_PD_DTYPE)
CPP_TYPE_TO_PD_DTYPE_REGISTER(PD_DTYPE_TO_CPP_TYPE)

namespace custom_kernel {

class DenseTensor {
 public:
  DenseTensor() : c_tensor(PD_NewTensor()), own_data(true) {}

  explicit DenseTensor(PD_Tensor* tensor) : c_tensor(tensor), own_data(false) {}

  ~DenseTensor() {
    if (own_data) {
      PD_DeleteTensor(c_tensor);
    }
  }

  std::vector<size_t> dims() const {
    C_Status status;
    auto dimension = PD_GetTensorDims(c_tensor, &status);
    PD_CHECK(status);
    return dimension;
  }

  PD_DataType dtype() const {
    C_Status status;
    auto data_type = PD_GetTensorType(c_tensor, &status);
    PD_CHECK(status);
    return data_type;
  }

  PD_DataLayout layout() const {
    C_Status status;
    auto data_layout = PD_GetTensorLayout(c_tensor, &status);
    PD_CHECK(status);
    return data_layout;
  }

  size_t numel() const {
    C_Status status;
    auto element_count = PD_GetTensorElementCount(c_tensor, &status);
    PD_CHECK(status);
    return element_count;
  }

  size_t memory_size() const {
    C_Status status;
    auto byte_size = PD_GetTensorByteSize(c_tensor, &status);
    PD_CHECK(status);
    return byte_size;
  }

  void set_dims(const std::vector<size_t>& dims) {
    C_Status status;
    PD_SetTensorDims(c_tensor, dims.size(), dims.data(), &status);
    PD_CHECK(status);
  }

  void set_dtype(PD_DataType data_type) {
    C_Status status;
    PD_SetTensorType(c_tensor, data_type, &status);
    PD_CHECK(status);
  }

  void set_dtype(PD_DataLayout data_layout) {
    C_Status status;
    PD_SetTensorLayout(c_tensor, data_layout, &status);
    PD_CHECK(status);
  }

  template <typename T>
  T* data() const {
    C_Status status;
    auto ptr = PD_GetTensorData(c_tensor, &status);
    PD_CHECK(status);
    return static_cast<T*>(ptr);
  }

  template <typename T>
  T* mutable_data(size_t size = 0, const PD_Context* ctx = nullptr) {
    if (size == 0) {
      return data<T>();
    }
    C_Status status;
    auto ptr = PD_AllocateTensor(
        ctx, c_tensor, size, phi::CppTypeToPDType<T>::Type(), &status);
    PD_CHECK(status);
    return static_cast<T*>(ptr);
  }

  void* mutable_data(PD_DataType data_type,
                     size_t size = 0,
                     const PD_Context* ctx = nullptr) {
    if (size == 0) {
      return data<void*>();
    }
    C_Status status;
    auto ptr = PD_AllocateTensor(ctx, c_tensor, size, data_type, &status);
    PD_CHECK(status);
    return static_cast<void*>(ptr);
  }

 private:
  PD_Tensor* c_tensor;
  bool own_data;
};

class DeviceContext {
 public:
  explicit DeviceContext(PD_Context* context) : c_context(context) {}

  void* stream() const {
    C_Status status;
    auto stream_ = PD_GetStream(c_context, &status);
    PD_CHECK(status);
    return stream_;
  }

  void* Alloc(DenseTensor* tensor,
              PD_DataType dtype,
              size_t requested_size = 0) const {
    return tensor->mutable_data(dtype, requested_size, c_context);
  }

  template <typename T>
  T* Alloc(DenseTensor* tensor, size_t requested_size = 0) const {
    return tensor->mutable_data<T>(requested_size, c_context);
  }

  void* HostAlloc(DenseTensor* tensor,
                  PD_DataType dtype,
                  size_t requested_size = 0) const {
    return tensor->mutable_data(dtype, requested_size);
  }

  template <typename T>
  T* HostAlloc(DenseTensor* tensor, size_t requested_size = 0) const {
    return tensor->mutable_data<T>(requested_size);
  }

 private:
  PD_Context* c_context;
};

class Scalar {
 public:
  explicit Scalar(PD_Scalar* scalar) : c_scalar(scalar) {}

  PD_DataType dtype() const { return PD_GetScalarType(c_scalar); }

  template <typename T>
  inline T to() const;

 private:
  PD_Scalar* c_scalar;
};

template <>
inline bool Scalar::to<bool>() const {
  return PD_GetScalarBoolData(c_scalar);
}

template <>
inline float Scalar::to<float>() const {
  return PD_GetScalarFloat32Data(c_scalar);
}

template <>
inline double Scalar::to<double>() const {
  return PD_GetScalarFloat64Data(c_scalar);
}

template <>
inline uint8_t Scalar::to<uint8_t>() const {
  return PD_GetScalarUInt8Data(c_scalar);
}

template <>
inline uint16_t Scalar::to<uint16_t>() const {
  return PD_GetScalarUInt16Data(c_scalar);
}

template <>
inline uint32_t Scalar::to<uint32_t>() const {
  return PD_GetScalarUInt32Data(c_scalar);
}

template <>
inline uint64_t Scalar::to<uint64_t>() const {
  return PD_GetScalarUInt64Data(c_scalar);
}

template <>
inline int8_t Scalar::to<int8_t>() const {
  return PD_GetScalarInt8Data(c_scalar);
}

template <>
inline int16_t Scalar::to<int16_t>() const {
  return PD_GetScalarInt16Data(c_scalar);
}

template <>
inline int32_t Scalar::to<int32_t>() const {
  return PD_GetScalarInt32Data(c_scalar);
}

template <>
inline int64_t Scalar::to<int64_t>() const {
  return PD_GetScalarInt64Data(c_scalar);
}

class IntArray {
 public:
  explicit IntArray(PD_IntArray* int_array) : c_int_array(int_array) {}

  size_t size() const { return PD_GetIntArraySize(c_int_array); }

  std::vector<int64_t> GetData() const {
    auto list = PD_GetIntArrayData(c_int_array);
    auto data = reinterpret_cast<int64_t*>(list.data);
    return std::vector<int64_t>(data, data + list.size);
  }

 private:
  PD_IntArray* c_int_array;
};

class Place {
 public:
  explicit Place(PD_Place* place) : c_place(place) {}

  bool is_host() { return PD_PlaceIsHost(c_place); }

  size_t GetDeviceID() { return PD_PlaceGetDeviceId(c_place); }

 private:
  PD_Place* c_place;
};

}  // namespace custom_kernel

using Context = custom_kernel::DeviceContext;
using DenseTensor = custom_kernel::DenseTensor;
using Scalar = custom_kernel::Scalar;
using IntArray = custom_kernel::IntArray;
using Place = custom_kernel::Place;
using DataType = ::PD_DataType;
using DataLayout = ::PD_DataLayout;

#define CPP_TYPE_TO_PD_ARG_TYPE_REGISTER(_)                                  \
  _(phi::custom_kernel::DenseTensor, PD_ArgumentType::PD_ARG_TYPE_TENSOR)    \
  _(phi::custom_kernel::DeviceContext, PD_ArgumentType::PD_ARG_TYPE_CONTEXT) \
  _(bool, PD_ArgumentType::PD_ARG_TYPE_BOOL)                                 \
  _(float, PD_ArgumentType::PD_ARG_TYPE_FLOAT32)                             \
  _(double, PD_ArgumentType::PD_ARG_TYPE_FLOAT64)                            \
  _(int32_t, PD_ArgumentType::PD_ARG_TYPE_INT32)                             \
  _(int64_t, PD_ArgumentType::PD_ARG_TYPE_INT64)                             \
  _(PD_DataType, PD_ArgumentType::PD_ARG_TYPE_DATA_TYPE)                     \
  _(PD_DataLayout, PD_ArgumentType::PD_ARG_TYPE_DATA_LAYOUT)                 \
  _(std::vector<int32_t>, PD_ArgumentType::PD_ARG_TYPE_LIST_INT32)           \
  _(std::vector<int64_t>, PD_ArgumentType::PD_ARG_TYPE_LIST_INT64)           \
  _(std::vector<float>, PD_ArgumentType::PD_ARG_TYPE_LIST_FLOAT32)           \
  _(std::vector<double>, PD_ArgumentType::PD_ARG_TYPE_LIST_FLOAT64)          \
  _(std::vector<bool>, PD_ArgumentType::PD_ARG_TYPE_LIST_BOOL)               \
  _(std::string, PD_ArgumentType::PD_ARG_TYPE_STRING)                        \
  _(phi::custom_kernel::Scalar, PD_ArgumentType::PD_ARG_TYPE_SCALAR)         \
  _(phi::custom_kernel::IntArray, PD_ArgumentType::PD_ARG_TYPE_INT_ARRAY)    \
  _(phi::custom_kernel::Place, PD_ArgumentType::PD_ARG_TYPE_PLACE)           \
  _(std::vector<std::string>, PD_ArgumentType::PD_ARG_TYPE_LIST_STRING)      \
  _(std::vector<phi::custom_kernel::Scalar>,                                 \
    PD_ArgumentType::PD_ARG_TYPE_LIST_SCALAR)

template <typename T>
struct CppTypeToPDArgumentType;

#define CPP_TYPE_TO_PD_ARG_TYPE(x, y)                     \
  template <>                                             \
  struct CppTypeToPDArgumentType<x> {                     \
    constexpr static PD_ArgumentType Type() { return y; } \
  };

template <PD_ArgumentType T>
struct PDArgumentTypeToCppType;

#define PD_ARG_TYPE_TO_CPP_TYPE(x, y) \
  template <>                         \
  struct PDArgumentTypeToCppType<y> { \
    using type = x;                   \
  };

CPP_TYPE_TO_PD_ARG_TYPE_REGISTER(CPP_TYPE_TO_PD_ARG_TYPE)
CPP_TYPE_TO_PD_ARG_TYPE_REGISTER(PD_ARG_TYPE_TO_CPP_TYPE)

}  // namespace phi

#include "paddle/phi/core/custom_phi_kernel_c_api_internal.h"

// clang-format off

#define PD_BUILD_PHI_KERNEL(kernel_name,                            \
                            backend,                                \
                            layout,                                 \
                            meta_kernel_fn,                         \
                            ...)                                    \
  template <typename kernel_type>                                   \
  struct __##kernel_name##_##backend##_##layout##__ {               \
    __##kernel_name##_##backend##_##layout##__() {                  \
      phi::CustomKernelArgsParseFunctor<decltype(                   \
          &meta_kernel_fn<kernel_type>)>                            \
          parser;                                                   \
      PD_RegisterPhiKernel(                                         \
          #kernel_name,                                             \
          #backend,                                                 \
          ::phi::CppTypeToPDType<kernel_type>::Type(),              \
          PD_DATALAYOUT(layout),                                    \
          parser.in_args_type.size(),                               \
          parser.in_args_type.data(),                               \
          parser.attr_args_type.size(),                             \
          parser.attr_args_type.data(),                             \
          parser.out_args_type.size(),                              \
          parser.out_args_type.data(),                              \
          CUSTOM_PHI_KERNEL(meta_kernel_fn<kernel_type>),           \
          CUSTOM_PHI_VARIADIC_KERNEL(                               \
            meta_kernel_fn<kernel_type>));                          \
    }                                                               \
    static void Touch() {}                                          \
  };                                                                \
  PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(              \
      CUSTOM_tp_ns_check_##kernel_name##_##backend##_##layout,      \
      "PD_BUILD_KERNEL must be called in global namespace.");       \
  static void                                                       \
      __CUSTOM_args_def_FN_##kernel_name##_##backend##_##layout(    \
          const PD_KernelKey* kernel_key, PD_Kernel* kernel);       \
  _PD_BUILD_PHI_KERNEL(__##kernel_name##_##backend##_##layout##__,  \
                       kernel_name,                                 \
                       backend,                                     \
                       layout,                                      \
                       meta_kernel_fn,                              \
                       __VA_ARGS__)                                 \
  void                                                              \
      __CUSTOM_args_def_FN_##kernel_name##_##backend##_##layout(    \
          const PD_KernelKey* kernel_key, PD_Kernel* kernel)

// clang-format on

#endif
