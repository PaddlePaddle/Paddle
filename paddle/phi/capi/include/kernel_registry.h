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

#include "paddle/phi/capi/include/wrapper_base.h"

namespace phi {
namespace capi {
// eager mode
inline std::vector<phi::capi::DenseTensor> PD_TensorVector(PD_Tensor *tensor) {
  std::vector<phi::capi::DenseTensor> ret;
  auto list = PD_TensorVectorToList(tensor);
  auto data = reinterpret_cast<PD_Tensor **>(list.data);
  for (size_t i = 0; i < list.size; ++i) {
    ret.emplace_back(data[i]);
  }
  return ret;
}

inline paddle::optional<phi::capi::DenseTensor> PD_OptionalTensor(
    PD_Tensor *tensor) {
  auto ptr = PD_OptionalTensorGetPointer(tensor);
  return ptr ? paddle::optional<phi::capi::DenseTensor>(
                   phi::capi::DenseTensor(ptr))
             : paddle::optional<phi::capi::DenseTensor>(paddle::none);
}

template <typename T>
inline T PD_Attr(void *attr) {
  return *reinterpret_cast<T *>(attr);
}

template <>
inline std::string PD_Attr<std::string>(void *attr) {
  return PD_StringAttr(attr);
}

template <>
inline PD_DataType PD_Attr<PD_DataType>(void *attr) {
  return PD_DatatTypeAttr(attr);
}

template <>
inline PD_DataLayout PD_Attr<PD_DataLayout>(void *attr) {
  return PD_DatatLayoutAttr(attr);
}

template <>
inline std::vector<int32_t> PD_Attr<std::vector<int32_t>>(void *attr) {
  auto list = PD_ListInt32Attr(attr);
  auto data = reinterpret_cast<int32_t *>(list.data);
  std::vector<int32_t> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<int64_t> PD_Attr<std::vector<int64_t>>(void *attr) {
  auto list = PD_ListInt64Attr(attr);
  auto data = reinterpret_cast<int64_t *>(list.data);
  std::vector<int64_t> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<float> PD_Attr<std::vector<float>>(void *attr) {
  auto list = PD_ListFloatAttr(attr);
  auto data = reinterpret_cast<float *>(list.data);
  std::vector<float> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<double> PD_Attr<std::vector<double>>(void *attr) {
  auto list = PD_ListDoubleAttr(attr);
  auto data = reinterpret_cast<double *>(list.data);
  std::vector<double> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline phi::capi::Scalar PD_Attr<phi::capi::Scalar>(void *attr) {
  return phi::capi::Scalar(reinterpret_cast<PD_Scalar *>(attr));
}

template <>
inline phi::capi::IntArray PD_Attr<phi::capi::IntArray>(void *attr) {
  return phi::capi::IntArray(reinterpret_cast<PD_IntArray *>(attr));
}

template <>
inline phi::capi::Place PD_Attr<phi::capi::Place>(void *attr) {
  return phi::capi::Place(reinterpret_cast<PD_Place *>(attr));
}

template <>
inline std::vector<phi::capi::Scalar> PD_Attr<std::vector<phi::capi::Scalar>>(
    void *attr) {
  auto c_list = PD_ListScalarAttr(attr);
  auto data = reinterpret_cast<PD_Scalar **>(c_list.data);
  std::vector<phi::capi::Scalar> list;
  for (size_t i = 0; i < c_list.size; ++i) {
    list.emplace_back(data[i]);
  }
  PD_DeletePointerList(c_list);
  return list;
}

template <>
inline std::vector<std::string> PD_Attr<std::vector<std::string>>(void *attr) {
  auto c_list = PD_ListStringAttr(attr);
  auto data = reinterpret_cast<char **>(c_list.data);
  std::vector<std::string> list;
  for (size_t i = 0; i < c_list.size; ++i) {
    list.emplace_back(data[i]);
  }
  PD_DeletePointerList(c_list);
  return list;
}

template <>
inline std::vector<bool> PD_Attr<std::vector<bool>>(void *attr) {
  auto c_list = PD_ListBoolAttr(attr);
  std::vector<bool> list;
  auto data = reinterpret_cast<uint8_t *>(c_list.data);
  for (size_t i = 0; i < c_list.size; ++i) {
    list[i] = static_cast<bool>(data[i]);
  }
  PD_DeleteUInt8List(c_list);
  return list;
}
//
inline phi::capi::DeviceContext PD_GetDeviceContext(PD_KernelContext *ctx) {
  return phi::capi::DeviceContext(PD_KernelContextGetDeviceContext(ctx));
}

inline phi::capi::DenseTensor PD_InputAt(PD_KernelContext *ctx, size_t index) {
  return phi::capi::DenseTensor(PD_KernelContextInputAt(ctx, index));
}

inline paddle::optional<phi::capi::DenseTensor> PD_OptionalInputAt(
    PD_KernelContext *ctx, size_t index) {
  auto tensor = PD_KernelContextInputAt(ctx, index);
  return tensor
             ? paddle::optional<phi::capi::DenseTensor>(phi::capi::DenseTensor(
                   reinterpret_cast<PD_Tensor *>(tensor)))
             : paddle::optional<phi::capi::DenseTensor>(paddle::none);
}

inline std::vector<phi::capi::DenseTensor> PD_MultiInputAt(
    PD_KernelContext *ctx, size_t index) {
  std::vector<phi::capi::DenseTensor> ret;
  auto list = PD_KernelContextMultiInputAt(ctx, index);
  auto data = reinterpret_cast<PD_Tensor **>(list.data);
  for (size_t i = 0; i < list.size; ++i) {
    ret.emplace_back(data[i]);
  }
  PD_DeletePointerList(list);
  return ret;
}

inline phi::capi::DenseTensor PD_OutputAt(PD_KernelContext *ctx, size_t index) {
  return phi::capi::DenseTensor(PD_KernelContextOutputAt(ctx, index));
}

inline std::vector<phi::capi::DenseTensor> PD_MultiOutputAt(
    PD_KernelContext *ctx, size_t index) {
  std::vector<phi::capi::DenseTensor> ret;
  auto list = PD_KernelContextMultiOutputAt(ctx, index);
  auto data = reinterpret_cast<PD_Tensor **>(list.data);
  for (size_t i = 0; i < list.size; ++i) {
    ret.emplace_back(data[i]);
  }
  PD_DeletePointerList(list);
  return ret;
}

template <typename T>
inline std::vector<T *> PD_GetPointerVector(std::vector<T> *vec) {
  std::vector<T *> ret;
  for (auto &item : *vec) {
    ret.push_back(&item);
  }
  return ret;
}

template <typename T>
inline T PD_AttrAt(PD_KernelContext *ctx, size_t index);

template <>
inline bool PD_AttrAt<bool>(PD_KernelContext *ctx, size_t index) {
  return PD_KernelContextBoolAttrAt(ctx, index);
}

template <>
inline int32_t PD_AttrAt<int32_t>(PD_KernelContext *ctx, size_t index) {
  return PD_KernelContextInt32AttrAt(ctx, index);
}

template <>
inline int64_t PD_AttrAt<int64_t>(PD_KernelContext *ctx, size_t index) {
  return PD_KernelContextInt64AttrAt(ctx, index);
}

template <>
inline float PD_AttrAt<float>(PD_KernelContext *ctx, size_t index) {
  return PD_KernelContextFloatAttrAt(ctx, index);
}

template <>
inline double PD_AttrAt<double>(PD_KernelContext *ctx, size_t index) {
  return PD_KernelContextDoubleAttrAt(ctx, index);
}

template <>
inline std::string PD_AttrAt<std::string>(PD_KernelContext *ctx, size_t index) {
  return PD_KernelContextStringAttrAt(ctx, index);
}

template <>
inline PD_DataType PD_AttrAt<PD_DataType>(PD_KernelContext *ctx, size_t index) {
  return PD_KernelContextDataTypeAttrAt(ctx, index);
}

template <>
inline PD_DataLayout PD_AttrAt<PD_DataLayout>(PD_KernelContext *ctx,
                                              size_t index) {
  return PD_KernelContextDataLayoutAttrAt(ctx, index);
}

template <>
inline std::vector<int32_t> PD_AttrAt<std::vector<int32_t>>(
    PD_KernelContext *ctx, size_t index) {
  auto list = PD_KernelContextListInt32AttrAt(ctx, index);
  auto data = reinterpret_cast<int32_t *>(list.data);
  std::vector<int32_t> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<int64_t> PD_AttrAt<std::vector<int64_t>>(
    PD_KernelContext *ctx, size_t index) {
  auto list = PD_KernelContextListInt64AttrAt(ctx, index);
  auto data = reinterpret_cast<int64_t *>(list.data);
  std::vector<int64_t> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<float> PD_AttrAt<std::vector<float>>(PD_KernelContext *ctx,
                                                        size_t index) {
  auto list = PD_KernelContextListFloatAttrAt(ctx, index);
  auto data = reinterpret_cast<float *>(list.data);
  std::vector<float> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<double> PD_AttrAt<std::vector<double>>(PD_KernelContext *ctx,
                                                          size_t index) {
  auto list = PD_KernelContextListDoubleAttrAt(ctx, index);
  auto data = reinterpret_cast<double *>(list.data);
  std::vector<double> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline phi::capi::Scalar PD_AttrAt<phi::capi::Scalar>(PD_KernelContext *ctx,
                                                      size_t index) {
  auto scalar = PD_KernelContextScalarAttrAt(ctx, index);
  return phi::capi::Scalar(scalar);
}

template <>
inline phi::capi::IntArray PD_AttrAt<phi::capi::IntArray>(PD_KernelContext *ctx,
                                                          size_t index) {
  auto int_array = PD_KernelContextIntArrayAttrAt(ctx, index);
  return phi::capi::IntArray(int_array);
}

template <>
inline phi::capi::Place PD_AttrAt<phi::capi::Place>(PD_KernelContext *ctx,
                                                    size_t index) {
  auto place = PD_KernelContextPlaceAttrAt(ctx, index);
  return phi::capi::Place(place);
}

template <>
inline std::vector<phi::capi::Scalar> PD_AttrAt<std::vector<phi::capi::Scalar>>(
    PD_KernelContext *ctx, size_t index) {
  auto c_list = PD_KernelContextListScalarAttrAt(ctx, index);
  auto data = reinterpret_cast<PD_Scalar **>(c_list.data);
  std::vector<phi::capi::Scalar> list;
  for (size_t i = 0; i < c_list.size; ++i) {
    list.emplace_back(data[i]);
  }
  PD_DeletePointerList(c_list);
  return list;
}

template <>
inline std::vector<std::string> PD_AttrAt<std::vector<std::string>>(
    PD_KernelContext *ctx, size_t index) {
  auto c_list = PD_KernelContextListStringAttrAt(ctx, index);
  auto data = reinterpret_cast<char **>(c_list.data);
  std::vector<std::string> list;
  for (size_t i = 0; i < c_list.size; ++i) {
    list.emplace_back(data[i]);
  }
  PD_DeletePointerList(c_list);
  return list;
}

template <>
inline std::vector<bool> PD_AttrAt<std::vector<bool>>(PD_KernelContext *ctx,
                                                      size_t index) {
  auto c_list = PD_KernelContextListBoolAttrAt(ctx, index);
  std::vector<bool> list;
  auto data = reinterpret_cast<uint8_t *>(c_list.data);
  for (size_t i = 0; i < c_list.size; ++i) {
    list[i] = static_cast<bool>(data[i]);
  }
  PD_DeleteUInt8List(c_list);
  return list;
}

#define CPP_TYPE_TO_PD_ARG_TYPE_REGISTER(_)                                 \
  _(phi::capi::DenseTensor, ::PD_KernelArgumentType::PD_ARG_TYPE_TENSOR)    \
  _(phi::capi::DeviceContext, ::PD_KernelArgumentType::PD_ARG_TYPE_CONTEXT) \
  _(bool, ::PD_KernelArgumentType::PD_ARG_TYPE_BOOL)                        \
  _(float, ::PD_KernelArgumentType::PD_ARG_TYPE_FLOAT32)                    \
  _(double, ::PD_KernelArgumentType::PD_ARG_TYPE_FLOAT64)                   \
  _(int32_t, ::PD_KernelArgumentType::PD_ARG_TYPE_INT32)                    \
  _(int64_t, ::PD_KernelArgumentType::PD_ARG_TYPE_INT64)                    \
  _(PD_DataType, ::PD_KernelArgumentType::PD_ARG_TYPE_DATA_TYPE)            \
  _(PD_DataLayout, ::PD_KernelArgumentType::PD_ARG_TYPE_DATA_LAYOUT)        \
  _(std::vector<int32_t>, ::PD_KernelArgumentType::PD_ARG_TYPE_LIST_INT32)  \
  _(std::vector<int64_t>, ::PD_KernelArgumentType::PD_ARG_TYPE_LIST_INT64)  \
  _(std::vector<float>, ::PD_KernelArgumentType::PD_ARG_TYPE_LIST_FLOAT32)  \
  _(std::vector<double>, ::PD_KernelArgumentType::PD_ARG_TYPE_LIST_FLOAT64) \
  _(std::vector<bool>, ::PD_KernelArgumentType::PD_ARG_TYPE_LIST_BOOL)      \
  _(std::string, ::PD_KernelArgumentType::PD_ARG_TYPE_STRING)               \
  _(phi::capi::Scalar, ::PD_KernelArgumentType::PD_ARG_TYPE_SCALAR)         \
  _(phi::capi::IntArray, ::PD_KernelArgumentType::PD_ARG_TYPE_INT_ARRAY)    \
  _(phi::capi::Place, ::PD_KernelArgumentType::PD_ARG_TYPE_PLACE)           \
  _(std::vector<std::string>,                                               \
    ::PD_KernelArgumentType::PD_ARG_TYPE_LIST_STRING)                       \
  _(std::vector<phi::capi::Scalar>,                                         \
    ::PD_KernelArgumentType::PD_ARG_TYPE_LIST_SCALAR)

template <typename T>
struct CppTypeToPDArgumentType;

#define CPP_TYPE_TO_PD_ARG_TYPE(x, y)                             \
  template <>                                                     \
  struct CppTypeToPDArgumentType<x> {                             \
    constexpr static ::PD_KernelArgumentType Type() { return y; } \
  };

template <::PD_KernelArgumentType T>
struct PDArgumentTypeToCppType;

#define PD_ARG_TYPE_TO_CPP_TYPE(x, y) \
  template <>                         \
  struct PDArgumentTypeToCppType<y> { \
    using type = x;                   \
  };

CPP_TYPE_TO_PD_ARG_TYPE_REGISTER(CPP_TYPE_TO_PD_ARG_TYPE)
CPP_TYPE_TO_PD_ARG_TYPE_REGISTER(PD_ARG_TYPE_TO_CPP_TYPE)

}  // namespace capi

using LoD = capi::LoD;
using Context = capi::DeviceContext;
using DenseTensor = capi::DenseTensor;
using Scalar = capi::Scalar;
using IntArray = capi::IntArray;
using Place = capi::Place;
using DataType = ::PD_DataType;
using DataLayout = ::PD_DataLayout;

}  // namespace phi

#include "paddle/phi/capi/include/kernel_utils.h"

// clang-format off

#define PD_BUILD_PHI_KERNEL(kernel_name,                            \
                            backend,                                \
                            layout,                                 \
                            meta_kernel_fn,                         \
                            ...)                                    \
  static void                                                       \
      __CUSTOM_adefs_CFN_##kernel_name##_##backend##_##layout(      \
          const PD_KernelKey* kernel_key, PD_Kernel* kernel);       \
  template <typename kernel_type>                                   \
  struct __##kernel_name##_##backend##_##layout##__ {               \
    __##kernel_name##_##backend##_##layout##__() {                  \
      ::phi::capi::CustomKernelArgsParseFunctor<decltype(           \
          &meta_kernel_fn<kernel_type>)>                            \
          parser;                                                   \
      PD_RegisterPhiKernel(                                         \
          #kernel_name,                                             \
          #backend,                                                 \
          ::phi::capi::CppTypeToPDType<kernel_type>::Type(),        \
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
          const ::phi::capi::KernelKey &kernel_key,                 \
          ::phi::capi::Kernel* kernel);                             \
  _PD_BUILD_PHI_KERNEL(__##kernel_name##_##backend##_##layout##__,  \
                       kernel_name,                                 \
                       backend,                                     \
                       layout,                                      \
                       meta_kernel_fn,                              \
                       __VA_ARGS__)                                 \
  void                                                              \
      __CUSTOM_adefs_CFN_##kernel_name##_##backend##_##layout(      \
          const PD_KernelKey* kernel_key, PD_Kernel* kernel) {      \
          auto cc_kernel = ::phi::capi::Kernel(kernel);             \
          __CUSTOM_adefs_FN_##kernel_name##_##backend##_##layout(   \
            ::phi::capi::KernelKey(                                 \
              const_cast<PD_KernelKey*>(kernel_key)),               \
            &cc_kernel);                                            \
      }                                                             \
  void                                                              \
      __CUSTOM_adefs_FN_##kernel_name##_##backend##_##layout(       \
          const ::phi::capi::KernelKey &kernel_key,                 \
          ::phi::capi::Kernel* kernel)

// clang-format on

#endif
