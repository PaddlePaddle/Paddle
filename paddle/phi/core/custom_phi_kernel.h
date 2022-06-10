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

#include <memory>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/custom_phi_kernel_c_api.h"
#include "paddle/utils/optional.h"

#define PD_CHECK_STATUS(status) PD_CHECK(status == C_SUCCESS)

inline std::vector<int64_t> PD_TensorGetDims(PD_Tensor* tensor,
                                             PD_Status* status) {
  int64_t ndims = PD_TensorGetNumDims(tensor, status);
  if (ndims > 0) {
    std::vector<int64_t> shape(ndims);
    for (int64_t i = 0; i < ndims; ++i) {
      shape[i] = PD_TensorGetDim(tensor, i, status);
    }
    return shape;
  }
  return std::vector<int64_t>();
}

namespace phi {

using LoD = std::vector<std::vector<size_t>>;

// utils

#define CPP_TYPE_TO_PD_DTYPE_REGISTER(_)         \
  _(bool, PD_DataType::BOOL)                     \
  _(phi::dtype::bfloat16, PD_DataType::BFLOAT16) \
  _(phi::dtype::float16, PD_DataType::FLOAT16)   \
  _(float, PD_DataType::FLOAT32)                 \
  _(double, PD_DataType::FLOAT64)                \
  _(uint8_t, PD_DataType::UINT8)                 \
  _(uint16_t, PD_DataType::UINT16)               \
  _(uint32_t, PD_DataType::UINT32)               \
  _(uint64_t, PD_DataType::UINT64)               \
  _(int8_t, PD_DataType::INT8)                   \
  _(int16_t, PD_DataType::INT16)                 \
  _(int32_t, PD_DataType::INT32)                 \
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

template <typename T>
class WrapperBase {
 public:
  explicit WrapperBase(T* ptr, bool own = false) : data_(ptr), own_(own) {}

  inline T* raw_data() const { return data_; }

  inline bool own_data() const { return own_; }

  inline void reset(const T* ptr) { data_ = ptr; }

 private:
  T* data_;
  bool own_;
};

class DenseTensor : public WrapperBase<PD_Tensor> {
 public:
  DenseTensor() : WrapperBase(PD_NewTensor(), true) {}

  explicit DenseTensor(PD_Tensor* tensor) : WrapperBase(tensor, false) {}

  ~DenseTensor() {
    if (own_data()) {
      PD_DeleteTensor(raw_data());
    }
  }

  bool valid() const {
    C_Status status;
    auto ret = PD_TensorIsValid(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return ret;
  }

  bool initialized() const {
    C_Status status;
    auto ret = PD_TensorIsInitialized(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return ret;
  }

  void* Holder() const {
    C_Status status;
    auto holder = PD_TensorGetHolder(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return holder;
  }

  std::vector<int64_t> dims() const {
    C_Status status;
    auto dimension = PD_TensorGetDims(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return dimension;
  }

  PD_DataType dtype() const {
    C_Status status;
    auto data_type = PD_TensorGetDataType(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return data_type;
  }

  PD_DataLayout layout() const {
    C_Status status;
    auto data_layout = PD_TensorGetDataLayout(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return data_layout;
  }

  int64_t numel() const {
    C_Status status;
    auto element_count = PD_TensorGetElementCount(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return element_count;
  }

  int64_t memory_size() const {
    C_Status status;
    auto byte_size = PD_TensorGetByteSize(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return byte_size;
  }

  LoD lod() const {
    PD_List data, offset;
    C_Status status;
    PD_TensorGetLoD(raw_data(), &data, &offset, &status);
    PD_CHECK_STATUS(status);
    LoD lod_;
    auto ptr = static_cast<size_t*>(data.data);
    auto offset_ptr = static_cast<size_t*>(offset.data);
    for (size_t i = 0; i < offset.size - 1; ++i) {
      lod_.emplace_back(ptr + offset_ptr[i], ptr + offset_ptr[i + 1]);
    }
    delete[] ptr;
    delete[] offset_ptr;
    return lod_;
  }

  void ResetLoD(const LoD& lod) {
    std::vector<size_t> data, offset;
    offset.push_back(0);
    for (const auto& item : lod) {
      data.insert(data.cend(), item.cbegin(), item.cend());
      offset.push_back(item.size());
    }
    PD_List data_list, offset_list;
    data_list = PDListFromVector(&data);
    offset_list = PDListFromVector(&offset);

    C_Status status;
    PD_TensorResetLoD(raw_data(), data_list, offset_list, &status);
    PD_CHECK_STATUS(status);
  }

  void Resize(const std::vector<int64_t>& dims) {
    C_Status status;
    PD_TensorSetDims(raw_data(), dims.size(), dims.data(), &status);
    PD_CHECK_STATUS(status);
  }

  void set_dtype(PD_DataType data_type) {
    C_Status status;
    PD_TensorSetDataType(raw_data(), data_type, &status);
    PD_CHECK_STATUS(status);
  }

  void set_layout(PD_DataLayout data_layout) {
    C_Status status;
    PD_TensorSetDataLayout(raw_data(), data_layout, &status);
    PD_CHECK_STATUS(status);
  }

  template <typename T>
  T* data() const {
    C_Status status;
    auto ptr = PD_TensorGetDataPointer(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return static_cast<T*>(ptr);
  }

  template <typename T>
  T* mutable_data(int64_t size = 0, const PD_DeviceContext* ctx = nullptr) {
    C_Status status;
    auto ptr = PD_DeviceContextAllocateTensor(
        ctx, raw_data(), size, phi::CppTypeToPDType<T>::Type(), &status);
    PD_CHECK_STATUS(status);
    return static_cast<T*>(ptr);
  }

  void* mutable_data(PD_DataType data_type,
                     int64_t size = 0,
                     const PD_DeviceContext* ctx = nullptr) {
    C_Status status;
    auto ptr = PD_DeviceContextAllocateTensor(
        ctx, raw_data(), size, data_type, &status);
    PD_CHECK_STATUS(status);
    return static_cast<void*>(ptr);
  }

  DenseTensor& ShareDataWith(const DenseTensor& src) {
    C_Status status;
    PD_TensorShareDataWith(raw_data(), src.raw_data(), &status);
    PD_CHECK_STATUS(status);
    return *this;
  }

  void share_lod(const DenseTensor& src) {
    C_Status status;
    PD_TensorShareLoDWith(raw_data(), src.raw_data(), &status);
    PD_CHECK_STATUS(status);
  }
};

class DeviceContext : public WrapperBase<PD_DeviceContext> {
 public:
  explicit DeviceContext(PD_DeviceContext* context)
      : WrapperBase<PD_DeviceContext>(context) {}

  void* stream() const {
    C_Status status;
    auto stream_ = PD_DeviceContextGetStream(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return stream_;
  }

  void* Alloc(DenseTensor* tensor,
              PD_DataType dtype,
              int64_t requested_size = 0) const {
    return tensor->mutable_data(dtype, requested_size, raw_data());
  }

  template <typename T>
  T* Alloc(DenseTensor* tensor, int64_t requested_size = 0) const {
    return tensor->mutable_data<T>(requested_size, raw_data());
  }

  void* HostAlloc(DenseTensor* tensor,
                  PD_DataType dtype,
                  int64_t requested_size = 0) const {
    return tensor->mutable_data(dtype, requested_size);
  }

  template <typename T>
  T* HostAlloc(DenseTensor* tensor, int64_t requested_size = 0) const {
    return tensor->mutable_data<T>(requested_size);
  }
};

class Scalar : public WrapperBase<PD_Scalar> {
 public:
  explicit Scalar(PD_Scalar* scalar) : WrapperBase<PD_Scalar>(scalar) {}

  PD_DataType dtype() const { return PD_ScalarGetDataType(raw_data()); }

  template <typename T>
  inline T to() const;
};

template <>
inline bool Scalar::to<bool>() const {
  return PD_ScalarGetBoolData(raw_data());
}

template <>
inline float Scalar::to<float>() const {
  return PD_ScalarGetFloat32Data(raw_data());
}

template <>
inline double Scalar::to<double>() const {
  return PD_ScalarGetFloat64Data(raw_data());
}

template <>
inline uint8_t Scalar::to<uint8_t>() const {
  return PD_ScalarGetUInt8Data(raw_data());
}

template <>
inline uint16_t Scalar::to<uint16_t>() const {
  return PD_ScalarGetUInt16Data(raw_data());
}

template <>
inline uint32_t Scalar::to<uint32_t>() const {
  return PD_ScalarGetUInt32Data(raw_data());
}

template <>
inline uint64_t Scalar::to<uint64_t>() const {
  return PD_ScalarGetUInt64Data(raw_data());
}

template <>
inline int8_t Scalar::to<int8_t>() const {
  return PD_ScalarGetInt8Data(raw_data());
}

template <>
inline int16_t Scalar::to<int16_t>() const {
  return PD_ScalarGetInt16Data(raw_data());
}

template <>
inline int32_t Scalar::to<int32_t>() const {
  return PD_ScalarGetInt32Data(raw_data());
}

template <>
inline int64_t Scalar::to<int64_t>() const {
  return PD_ScalarGetInt64Data(raw_data());
}

class IntArray : WrapperBase<PD_IntArray> {
 public:
  explicit IntArray(PD_IntArray* int_array)
      : WrapperBase<PD_IntArray>(int_array) {}

  size_t size() const { return PD_IntArrayGetElementCount(raw_data()); }

  std::vector<int64_t> GetData() const {
    auto list = PD_IntArrayGetDataPointer(raw_data());
    auto data = reinterpret_cast<int64_t*>(list.data);
    std::vector<int64_t> ret(data, data + list.size);
    return ret;
  }
};

class Place : WrapperBase<PD_Place> {
 public:
  explicit Place(PD_Place* place) : WrapperBase<PD_Place>(place) {}

  bool is_host() { return PD_PlaceIsHost(raw_data()); }

  int8_t GetDeviceID() { return PD_PlaceGetDeviceId(raw_data()); }
};

class TensorArgDef : WrapperBase<PD_TensorArgDef> {
 public:
  explicit TensorArgDef(PD_TensorArgDef* tensor_arg_def)
      : WrapperBase<PD_TensorArgDef>(tensor_arg_def) {}

  // TensorArgDef& SetBackend() {
  //   return *this;
  // }

  TensorArgDef& SetDataLayout(PD_DataLayout in_layout) {
    C_Status status;
    PD_TensorArgDefSetDataLayout(raw_data(), in_layout, &status);
    PD_CHECK_STATUS(status);
    return *this;
  }

  TensorArgDef& SetDataType(PD_DataType in_dtype) {
    C_Status status;
    PD_TensorArgDefSetDataType(raw_data(), in_dtype, &status);
    PD_CHECK_STATUS(status);
    return *this;
  }
};

class KernelArgsDef : WrapperBase<PD_KernelArgsDef> {
 public:
  explicit KernelArgsDef(PD_KernelArgsDef* kernel_args_def)
      : WrapperBase<PD_KernelArgsDef>(kernel_args_def) {}

  std::vector<TensorArgDef> input_defs() {
    C_Status status;
    auto list = PD_KernelArgsDefGetInputArgDefs(raw_data(), &status);
    PD_CHECK_STATUS(status);
    auto ptr = reinterpret_cast<PD_TensorArgDef**>(list.data);
    std::vector<TensorArgDef> ret;
    for (size_t i = 0; i < list.size; ++i) {
      ret.emplace_back(ptr[i]);
    }
    PD_DeleteList(list);
    return ret;
  }

  std::vector<TensorArgDef> output_defs() {
    C_Status status;
    auto list = PD_KernelArgsDefGetOutputArgDefs(raw_data(), &status);
    PD_CHECK_STATUS(status);
    auto ptr = reinterpret_cast<PD_TensorArgDef**>(list.data);
    std::vector<TensorArgDef> ret;
    for (size_t i = 0; i < list.size; ++i) {
      ret.emplace_back(ptr[i]);
    }
    PD_DeleteList(list);
    return ret;
  }

  // std::vector<AttributeArgDef>
  // attribute_defs() {
  // }
};

class KernelKey : WrapperBase<PD_KernelKey> {
 public:
  explicit KernelKey(PD_KernelKey* kernel_key)
      : WrapperBase<PD_KernelKey>(kernel_key) {}

  // Backend backend() const { return backend_; }
  PD_DataLayout layout() const {
    PD_Status status;
    auto layout_ = PD_KernelKeyGetLayout(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return layout_;
  }

  PD_DataType dtype() const {
    PD_Status status;
    auto dtype_ = PD_KernelKeyGetDataType(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return dtype_;
  }
};

class Kernel : WrapperBase<PD_Kernel> {
 public:
  explicit Kernel(PD_Kernel* kernel) : WrapperBase<PD_Kernel>(kernel) {}

  KernelArgsDef args_def() const {
    C_Status status;
    auto ptr = PD_KernelGetArgsDef(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return KernelArgsDef(ptr);
  }

  TensorArgDef InputAt(size_t idx) { return args_def().input_defs()[idx]; }

  TensorArgDef OutputAt(size_t idx) { return args_def().input_defs()[idx]; }
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
  static void                                                       \
      __CUSTOM_adefs_CFN_##kernel_name##_##backend##_##layout(   \
          const PD_KernelKey* kernel_key, PD_Kernel* kernel);       \
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
          __CUSTOM_adefs_CFN_##kernel_name##_##backend##_##layout,  \
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
      __CUSTOM_adefs_FN_##kernel_name##_##backend##_##layout(       \
          const ::phi::custom_kernel::KernelKey &kernel_key,        \
          ::phi::custom_kernel::Kernel* kernel);                    \
  _PD_BUILD_PHI_KERNEL(__##kernel_name##_##backend##_##layout##__,  \
                       kernel_name,                                 \
                       backend,                                     \
                       layout,                                      \
                       meta_kernel_fn,                              \
                       __VA_ARGS__)                                 \
  void                                                              \
      __CUSTOM_adefs_CFN_##kernel_name##_##backend##_##layout(      \
          const PD_KernelKey* kernel_key, PD_Kernel* kernel) {      \
          auto cc_kernel = ::phi::custom_kernel::Kernel(kernel);    \
          __CUSTOM_adefs_FN_##kernel_name##_##backend##_##layout(   \
            ::phi::custom_kernel::KernelKey(                        \
              const_cast<PD_KernelKey*>(kernel_key)),               \
            &cc_kernel);                                            \
      }                                                             \
  void                                                              \
      __CUSTOM_adefs_FN_##kernel_name##_##backend##_##layout(       \
          const ::phi::custom_kernel::KernelKey &kernel_key,        \
          ::phi::custom_kernel::Kernel* kernel)

// clang-format on

#endif
