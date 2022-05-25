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

// cpp wrapper
inline phi::custom_kernel::DeviceContext PD_GetContext(
    PD_ExecutionContext *ctx) {
  return phi::custom_kernel::DeviceContext(PD_OriginGetContext(ctx));
}

inline phi::custom_kernel::DenseTensor PD_InputAt(PD_ExecutionContext *ctx,
                                                  size_t index) {
  return phi::custom_kernel::DenseTensor(PD_OriginInputAt(ctx, index));
}

inline paddle::optional<phi::custom_kernel::DenseTensor> PD_OptionalInputAt(
    PD_ExecutionContext *ctx, size_t index) {
  auto tensor = PD_OriginOptionalInputAt(ctx, index);
  return tensor
             ? paddle::optional<phi::custom_kernel::DenseTensor>(
                   phi::custom_kernel::DenseTensor(tensor))
             : paddle::optional<phi::custom_kernel::DenseTensor>(paddle::none);
}

inline std::vector<phi::custom_kernel::DenseTensor> PD_MultiInputAt(
    PD_ExecutionContext *ctx, size_t index) {
  std::vector<phi::custom_kernel::DenseTensor> ret;
  auto list = PD_OriginMultiInputAt(ctx, index);
  auto data = reinterpret_cast<PD_Tensor **>(list.data);
  for (size_t i = 0; i < list.size; ++i) {
    ret.emplace_back(data[i]);
  }
  return ret;
}

inline phi::custom_kernel::DenseTensor PD_OutputAt(PD_ExecutionContext *ctx,
                                                   size_t index) {
  return phi::custom_kernel::DenseTensor(PD_OriginOutputAt(ctx, index));
}

inline std::vector<phi::custom_kernel::DenseTensor> PD_MultiOutputAt(
    PD_ExecutionContext *ctx, size_t index) {
  std::vector<phi::custom_kernel::DenseTensor> ret;
  auto list = PD_OriginMultiOutputAt(ctx, index);
  auto data = reinterpret_cast<PD_Tensor **>(list.data);
  for (size_t i = 0; i < list.size; ++i) {
    ret.emplace_back(data[i]);
  }
  return ret;
}

template <typename T>
inline T PD_AttrAt(PD_ExecutionContext *ctx, size_t index);

template <>
inline bool PD_AttrAt<bool>(PD_ExecutionContext *ctx, size_t index) {
  return PD_BoolAttrAt(ctx, index);
}

template <>
inline int32_t PD_AttrAt<int32_t>(PD_ExecutionContext *ctx, size_t index) {
  return PD_Int32AttrAt(ctx, index);
}

template <>
inline int64_t PD_AttrAt<int64_t>(PD_ExecutionContext *ctx, size_t index) {
  return PD_Int64AttrAt(ctx, index);
}

template <>
inline float PD_AttrAt<float>(PD_ExecutionContext *ctx, size_t index) {
  return PD_FloatAttrAt(ctx, index);
}

template <>
inline double PD_AttrAt<double>(PD_ExecutionContext *ctx, size_t index) {
  return PD_DoubleAttrAt(ctx, index);
}

template <>
inline std::string PD_AttrAt<std::string>(PD_ExecutionContext *ctx,
                                          size_t index) {
  return PD_StringAttrAt(ctx, index);
}

template <>
inline PD_DataType PD_AttrAt<PD_DataType>(PD_ExecutionContext *ctx,
                                          size_t index) {
  return PD_DataTypeAttrAt(ctx, index);
}

template <>
inline PD_DataLayout PD_AttrAt<PD_DataLayout>(PD_ExecutionContext *ctx,
                                              size_t index) {
  return PD_DataLayoutAttrAt(ctx, index);
}

template <>
inline std::vector<int32_t> PD_AttrAt<std::vector<int32_t>>(
    PD_ExecutionContext *ctx, size_t index) {
  auto list = PD_ListInt32AttrAt(ctx, index);
  auto data = reinterpret_cast<int32_t *>(list.data);
  std::vector<int32_t> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<int64_t> PD_AttrAt<std::vector<int64_t>>(
    PD_ExecutionContext *ctx, size_t index) {
  auto list = PD_ListInt64AttrAt(ctx, index);
  auto data = reinterpret_cast<int64_t *>(list.data);
  std::vector<int64_t> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<float> PD_AttrAt<std::vector<float>>(
    PD_ExecutionContext *ctx, size_t index) {
  auto list = PD_ListFloatAttrAt(ctx, index);
  auto data = reinterpret_cast<float *>(list.data);
  std::vector<float> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline std::vector<double> PD_AttrAt<std::vector<double>>(
    PD_ExecutionContext *ctx, size_t index) {
  auto list = PD_ListDoubleAttrAt(ctx, index);
  auto data = reinterpret_cast<double *>(list.data);
  std::vector<double> cc_list(data, data + list.size);
  return cc_list;
}

template <>
inline phi::custom_kernel::Scalar PD_AttrAt<phi::custom_kernel::Scalar>(
    PD_ExecutionContext *ctx, size_t index) {
  auto scalar = PD_ScalarAttrAt(ctx, index);
  return phi::custom_kernel::Scalar(scalar);
}

template <>
inline phi::custom_kernel::IntArray PD_AttrAt<phi::custom_kernel::IntArray>(
    PD_ExecutionContext *ctx, size_t index) {
  auto int_array = PD_IntArrayAttrAt(ctx, index);
  return phi::custom_kernel::IntArray(int_array);
}

template <>
inline phi::custom_kernel::Place PD_AttrAt<phi::custom_kernel::Place>(
    PD_ExecutionContext *ctx, size_t index) {
  auto place = PD_PlaceAttrAt(ctx, index);
  return phi::custom_kernel::Place(place);
}

template <>
inline std::vector<phi::custom_kernel::Scalar>
PD_AttrAt<std::vector<phi::custom_kernel::Scalar>>(PD_ExecutionContext *ctx,
                                                   size_t index) {
  auto c_list = PD_ListScalarAttrAt(ctx, index);
  auto data = reinterpret_cast<PD_Scalar **>(c_list.data);
  std::vector<phi::custom_kernel::Scalar> list;
  for (size_t i = 0; i < c_list.size; ++i) {
    list.emplace_back(data[i]);
  }
  PD_DeleteList(c_list);
  return list;
}

template <>
inline std::vector<std::string> PD_AttrAt<std::vector<std::string>>(
    PD_ExecutionContext *ctx, size_t index) {
  auto c_list = PD_ListScalarAttrAt(ctx, index);
  auto data = reinterpret_cast<char **>(c_list.data);
  std::vector<std::string> list;
  for (size_t i = 0; i < c_list.size; ++i) {
    list.emplace_back(data[i]);
  }
  PD_DeleteList(c_list);
  return list;
}

template <>
inline std::vector<bool> PD_AttrAt<std::vector<bool>>(PD_ExecutionContext *ctx,
                                                      size_t index) {
  auto c_list = PD_ListBoolAttrAt(ctx, index);
  std::vector<bool> list;
  auto data = reinterpret_cast<uint8_t *>(c_list.data);
  for (size_t i = 0; i < c_list.size; ++i) {
    list[i] = static_cast<bool>(data[i]);
  }
  PD_DeleteUInt8List(c_list);
  return list;
}

template <typename T>
struct CustomKernelCppWrapper;

template <>
struct CustomKernelCppWrapper<PD_Tensor> {
  using type = phi::custom_kernel::DenseTensor;
};

template <>
struct CustomKernelCppWrapper<PD_Context> {
  using type = phi::custom_kernel::DeviceContext;
};

template <>
struct CustomKernelCppWrapper<PD_IntArray> {
  using type = phi::custom_kernel::IntArray;
};

template <>
struct CustomKernelCppWrapper<PD_Scalar> {
  using type = phi::custom_kernel::Scalar;
};

#define PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg) \
  _PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)

#define _PD_CUSTOM_PHI_KERNEL_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)  \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define CUSTOM_PHI_KERNEL(...) \
  ::phi::CustomKernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define CUSTOM_PHI_VARIADIC_KERNEL(...)                \
  reinterpret_cast<void *>(                            \
      &::phi::CustomKernelImpl<decltype(&__VA_ARGS__), \
                               &__VA_ARGS__>::VariadicCompute)

#define PD_CUSTOM_NARGS(...) \
  _PD_CUSTOM_NARGS((__VA_ARGS__, _PD_CUSTOM_RESQ_N()))
#define _PD_CUSTOM_NARGS(...) _PD_CUSTOM_ARG_N(__VA_ARGS__)
#define _PD_CUSTOM_ARG_N_EXPAND(                                              \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) \
  N
#define _PD_CUSTOM_ARG_N(args) _PD_CUSTOM_ARG_N_EXPAND args
#define _PD_CUSTOM_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define PD_DATALAYOUT(arg__) PD_DataLayout::arg__

#ifdef __COUNTER__
#define PD_CUSTOM_PHI_KERNEL_ID __COUNTER__
#else
#define PD_CUSTOM_PHI_KERNEL_ID __LINE__
#endif

#define PD_CUSTOM_PHI_KERNEL_CONCATENATE(arg1, arg2) \
  PD_CUSTOM_PHI_KERNEL_CONCATENATE1(arg1, arg2)
#define PD_CUSTOM_PHI_KERNEL_CONCATENATE1(arg1, arg2) \
  PD_CUSTOM_PHI_KERNEL_CONCATENATE2(arg1, arg2)
#define PD_CUSTOM_PHI_KERNEL_CONCATENATE2(arg1, arg2) arg1##arg2
#define PD_CUSTOM_PHI_KERNEL_EXPAND(x) x

#define _PD_BUILD_KERNEL_INSTANTIATION(N, meta_kernel_fn, backend, ...) \
  PD_CUSTOM_PHI_KERNEL_CONCATENATE(_PD_BUILD_KERNEL_INSTANTIATION_, N)  \
  (meta_kernel_fn, backend, __VA_ARGS__)

#define _PD_BUILD_KERNEL_INSTANTIATION_1(meta_kernel_fn, backend, cpp_dtype) \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>
#define _PD_BUILD_KERNEL_INSTANTIATION_2(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_1(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_3(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_2(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_4(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_3(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_5(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_4(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_6(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_5(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_7(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_6(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_8(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_7(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_9(                                 \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_8(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_10(                                \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_9(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_11(                                \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_10(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_12(                                \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_11(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_13(                                \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_12(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_14(                                \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_13(meta_kernel_fn, backend, __VA_ARGS__))
#define _PD_BUILD_KERNEL_INSTANTIATION_15(                                \
    meta_kernel_fn, backend, cpp_dtype, ...)                              \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                            \
      _PD_BUILD_KERNEL_INSTANTIATION_14(meta_kernel_fn, backend, __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_1(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype)                          \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  int TouchCustomKernelSymbolFor_##kernel_name##_##backend##_##layout() {     \
    return 0;                                                                 \
  }

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_2(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_1(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_3(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_2(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_4(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_3(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_5(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_4(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_6(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_5(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_7(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_6(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_8(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_7(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_9(registrar_class,                    \
                                          kernel_name,                        \
                                          backend,                            \
                                          layout,                             \
                                          registrar_id,                       \
                                          meta_kernel_fn,                     \
                                          cpp_dtype,                          \
                                          ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_8(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_10(registrar_class,                   \
                                           kernel_name,                       \
                                           backend,                           \
                                           layout,                            \
                                           registrar_id,                      \
                                           meta_kernel_fn,                    \
                                           cpp_dtype,                         \
                                           ...)                               \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_9(registrar_class,                      \
                                        kernel_name,                          \
                                        backend,                              \
                                        layout,                               \
                                        PD_CUSTOM_PHI_KERNEL_ID,              \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_11(registrar_class,                   \
                                           kernel_name,                       \
                                           backend,                           \
                                           layout,                            \
                                           registrar_id,                      \
                                           meta_kernel_fn,                    \
                                           cpp_dtype,                         \
                                           ...)                               \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_10(registrar_class,                     \
                                         kernel_name,                         \
                                         backend,                             \
                                         layout,                              \
                                         PD_CUSTOM_PHI_KERNEL_ID,             \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_12(registrar_class,                   \
                                           kernel_name,                       \
                                           backend,                           \
                                           layout,                            \
                                           registrar_id,                      \
                                           meta_kernel_fn,                    \
                                           cpp_dtype,                         \
                                           ...)                               \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_11(registrar_class,                     \
                                         kernel_name,                         \
                                         backend,                             \
                                         layout,                              \
                                         PD_CUSTOM_PHI_KERNEL_ID,             \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_13(registrar_class,                   \
                                           kernel_name,                       \
                                           backend,                           \
                                           layout,                            \
                                           registrar_id,                      \
                                           meta_kernel_fn,                    \
                                           cpp_dtype,                         \
                                           ...)                               \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_12(registrar_class,                     \
                                         kernel_name,                         \
                                         backend,                             \
                                         layout,                              \
                                         PD_CUSTOM_PHI_KERNEL_ID,             \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_14(registrar_class,                   \
                                           kernel_name,                       \
                                           backend,                           \
                                           layout,                            \
                                           registrar_id,                      \
                                           meta_kernel_fn,                    \
                                           cpp_dtype,                         \
                                           ...)                               \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_13(registrar_class,                     \
                                         kernel_name,                         \
                                         backend,                             \
                                         layout,                              \
                                         PD_CUSTOM_PHI_KERNEL_ID,             \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_15(registrar_class,                   \
                                           kernel_name,                       \
                                           backend,                           \
                                           layout,                            \
                                           registrar_id,                      \
                                           meta_kernel_fn,                    \
                                           cpp_dtype,                         \
                                           ...)                               \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(   \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_14(registrar_class,                     \
                                         kernel_name,                         \
                                         backend,                             \
                                         layout,                              \
                                         PD_CUSTOM_PHI_KERNEL_ID,             \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT(                                   \
    N, registrar_class, kernel_name, backend, layout, meta_kernel_fn, ...) \
  PD_CUSTOM_PHI_KERNEL_EXPAND(PD_CUSTOM_PHI_KERNEL_CONCATENATE(            \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_, N)(registrar_class,                \
                                           kernel_name,                    \
                                           backend,                        \
                                           layout,                         \
                                           PD_CUSTOM_PHI_KERNEL_ID,        \
                                           meta_kernel_fn,                 \
                                           __VA_ARGS__))

#define PD_BUILD_KERNEL_REGISTRAR_INIT(                                 \
    registrar_class, kernel_name, backend, layout, meta_kernel_fn, ...) \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                          \
      _PD_BUILD_KERNEL_REGISTRAR_INIT(PD_CUSTOM_NARGS(__VA_ARGS__),     \
                                      registrar_class,                  \
                                      kernel_name,                      \
                                      backend,                          \
                                      layout,                           \
                                      meta_kernel_fn,                   \
                                      __VA_ARGS__))

#define PD_BUILD_KERNEL_INSTANTIATION(meta_kernel_fn, backend, ...) \
  _PD_BUILD_KERNEL_INSTANTIATION(                                   \
      PD_CUSTOM_NARGS(__VA_ARGS__), meta_kernel_fn, backend, __VA_ARGS__)

#define _PD_BUILD_2TA_KERNEL(                                           \
    registrar_class, kernel_name, backend, layout, meta_kernel_fn, ...) \
  PD_BUILD_KERNEL_INSTANTIATION(meta_kernel_fn, backend, __VA_ARGS__);  \
  PD_BUILD_KERNEL_REGISTRAR_INIT(registrar_class,                       \
                                 kernel_name,                           \
                                 backend,                               \
                                 layout,                                \
                                 meta_kernel_fn,                        \
                                 __VA_ARGS__);

#define _PD_BUILD_PHI_KERNEL(                                           \
    registrar_class, kernel_name, backend, layout, meta_kernel_fn, ...) \
  PD_CUSTOM_PHI_KERNEL_EXPAND(_PD_BUILD_2TA_KERNEL(registrar_class,     \
                                                   kernel_name,         \
                                                   backend,             \
                                                   layout,              \
                                                   meta_kernel_fn,      \
                                                   __VA_ARGS__))

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_DEVICE_CONTEXT_C_API(dev_ctx) \
  template <typename... Tail>                                                  \
  struct CustomKernelCallHelper<const dev_ctx *, Tail...> {                    \
    template <int dev_ctx_idx,                                                 \
              int in_idx,                                                      \
              int attr_idx,                                                    \
              int out_idx,                                                     \
              typename... PreviousArgs>                                        \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) {   \
      static_assert(in_idx == 0,                                               \
                    "Kernel's DeviceContext should appear before Inputs.");    \
      static_assert(                                                           \
          attr_idx == 0,                                                       \
          "Kernel's DeviceContext should appear before Attributes.");          \
      static_assert(out_idx == 0,                                              \
                    "Kernel's DeviceContext should appear before Outputs.");   \
      const dev_ctx *arg = PD_OriginGetContext(ctx);                           \
      CustomKernelCallHelper<Tail...>::                                        \
          template Compute<dev_ctx_idx + 1, in_idx, attr_idx, out_idx>(        \
              ctx, pargs..., arg);                                             \
    }                                                                          \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_DEVICE_CONTEXT(dev_ctx)       \
  template <typename... Tail>                                                  \
  struct CustomKernelCallHelper<const CustomKernelCppWrapper<dev_ctx>::type &, \
                                Tail...> {                                     \
    template <int dev_ctx_idx,                                                 \
              int in_idx,                                                      \
              int attr_idx,                                                    \
              int out_idx,                                                     \
              typename... PreviousArgs>                                        \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) {   \
      static_assert(in_idx == 0,                                               \
                    "Kernel's DeviceContext should appear before Inputs.");    \
      static_assert(                                                           \
          attr_idx == 0,                                                       \
          "Kernel's DeviceContext should appear before Attributes.");          \
      static_assert(out_idx == 0,                                              \
                    "Kernel's DeviceContext should appear before Outputs.");   \
      dev_ctx *arg = PD_OriginGetContext(ctx);                                 \
      CustomKernelCppWrapper<dev_ctx>::type arg_wrapper(arg);                  \
      CustomKernelCallHelper<Tail...>::                                        \
          template Compute<dev_ctx_idx + 1, in_idx, attr_idx, out_idx>(        \
              ctx, pargs..., arg_wrapper);                                     \
    }                                                                          \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_INPUT_C_API(tensor_type)    \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<const tensor_type *, Tail...> {              \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) { \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      const tensor_type *arg = PD_OriginInputAt(ctx, in_idx);                \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_INPUT(tensor_type)          \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<                                             \
      const CustomKernelCppWrapper<tensor_type>::type &,                     \
      Tail...> {                                                             \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) { \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      tensor_type *arg = PD_OriginInputAt(ctx, in_idx);                      \
      CustomKernelCppWrapper<tensor_type>::type arg_wrapper(arg);            \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(      \
              ctx, pargs..., arg_wrapper);                                   \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_OPTIONAL_INPUT(tensor_type) \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<                                             \
      paddle::optional<const CustomKernelCppWrapper<tensor_type>::type &>,   \
      Tail...> {                                                             \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) { \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      auto *arg = PD_OriginOptionalInputAt(ctx, in_idx);                     \
      CustomKernelCppWrapper<tensor_type>::type tmp_arg(arg);                \
      paddle::optional<const CustomKernelCppWrapper<tensor_type>::type &>    \
      arg_wrapper(                                                           \
          arg ? static_cast<const paddle::optional<                          \
                    const CustomKernelCppWrapper<tensor_type>::type &>>(     \
                    tmp_arg)                                                 \
              : paddle::none);                                               \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(      \
              ctx, pargs..., arg_wrapper);                                   \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_OUTPUT_C_API(tensor_type)   \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<tensor_type *, Tail...> {                    \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) { \
      tensor_type *arg = PD_OriginOutputAt(ctx, out_idx);                    \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx, out_idx + 1>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_OUTPUT(tensor_type)         \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<CustomKernelCppWrapper<tensor_type>::type *, \
                                Tail...> {                                   \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) { \
      tensor_type *arg = PD_OriginOutputAt(ctx, out_idx);                    \
      CustomKernelCppWrapper<tensor_type>::type arg_wrapper(arg);            \
      CustomKernelCppWrapper<tensor_type>::type *arg_wrapper_ref =           \
          arg ? &arg_wrapper : nullptr;                                      \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx, out_idx + 1>(      \
              ctx, pargs..., arg_wrapper_ref);                               \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(attr_type)        \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<attr_type, Tail...> {                        \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) { \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Attributes should appear before Outputs.");    \
      attr_type arg = PD_AttrAt<attr_type>(ctx, attr_idx);                   \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx + 1, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(        \
    attr_type)                                                               \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<const attr_type &, Tail...> {                \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) { \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Attributes should appear before Outputs.");    \
      attr_type arg = PD_AttrAt<attr_type>(ctx, attr_idx);                   \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx + 1, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_INPUT(tensor_type)     \
  template <typename... Tail>                                                 \
  struct CustomKernelCallHelper<                                              \
      const std::vector<const CustomKernelCppWrapper<tensor_type>::type *> &, \
      Tail...> {                                                              \
    template <int dev_ctx_idx,                                                \
              int in_idx,                                                     \
              int attr_idx,                                                   \
              int out_idx,                                                    \
              typename... PreviousArgs>                                       \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) {  \
      static_assert(attr_idx == 0,                                            \
                    "Kernel's Input should appear before Attributes.");       \
      static_assert(out_idx == 0,                                             \
                    "Kernel's Input should appear before Outputs.");          \
      PD_List list = PD_OriginMultiInputAt(ctx, in_idx);                      \
      std::vector<CustomKernelCppWrapper<tensor_type>::type> arg;             \
      for (size_t i = 0; i < list.size; ++i) {                                \
        arg.push_back(CustomKernelCppWrapper<tensor_type>::type(              \
            reinterpret_cast<PD_Tensor **>(list.data)[i]));                   \
      }                                                                       \
      std::vector<const CustomKernelCppWrapper<tensor_type>::type *>          \
          arg_wrapper;                                                        \
      for (auto &item : arg) {                                                \
        arg_wrapper.push_back(&item);                                         \
      }                                                                       \
      CustomKernelCallHelper<Tail...>::                                       \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(       \
              ctx, pargs..., arg_wrapper);                                    \
    }                                                                         \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_OUTPUT(tensor_type)     \
  template <typename... Tail>                                                  \
  struct CustomKernelCallHelper<                                               \
      std::vector<CustomKernelCppWrapper<tensor_type>::type *>,                \
      Tail...> {                                                               \
    template <int dev_ctx_idx,                                                 \
              int in_idx,                                                      \
              int attr_idx,                                                    \
              int out_idx,                                                     \
              typename... PreviousArgs>                                        \
    static void Compute(PD_ExecutionContext *ctx, PreviousArgs &... pargs) {   \
      static_assert(attr_idx == 0,                                             \
                    "Kernel's Input should appear before Attributes.");        \
      static_assert(out_idx == 0,                                              \
                    "Kernel's Input should appear before Outputs.");           \
      PD_List list = PD_OriginMultiOutputAt(ctx, in_idx);                      \
      std::vector<CustomKernelCppWrapper<tensor_type>::type> arg;              \
      std::vector<const CustomKernelCppWrapper<tensor_type>::type *>           \
          arg_wrapper;                                                         \
      for (size_t i = 0; i < list.size; ++i) {                                 \
        arg.push_back(CustomKernelCppWrapper<tensor_type>::type(               \
            reinterpret_cast<PD_Tensor **>(list.data)[i]));                    \
        arg_wrapper.push_back(                                                 \
            reinterpret_cast<PD_Tensor **>(list.data)[i] ? &arg[i] : nullptr); \
      }                                                                        \
      CustomKernelCallHelper<Tail...>::                                        \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(        \
              ctx, pargs..., arg_wrapper);                                     \
    }                                                                          \
  }

namespace phi {

template <typename T>
struct CustomTypeTag {};

template <typename Fn, Fn fn>
struct CustomKernelImpl;

template <typename Return,
          typename DevCtx,
          typename... Args,
          Return (*kernel_fn)(DevCtx, Args...)>
struct CustomKernelImpl<Return (*)(DevCtx, Args...), kernel_fn> {
  static void Compute(PD_ExecutionContext *ctx) {
    CustomKernelCallHelper<DevCtx, Args..., CustomTypeTag<int>>::
        template Compute<0, 0, 0, 0>(ctx);
  }

  // static void VariadicCompute(const PD_Context *dev_ctx, Args... args) {
  //   return kernel_fn(static_cast<DevCtx>(dev_ctx),
  //   std::forward<Args>(args)...);
  // }

  static void VariadicCompute(const phi::custom_kernel::DeviceContext &dev_ctx,
                              Args... args) {
    return kernel_fn(static_cast<DevCtx>(dev_ctx), std::forward<Args>(args)...);
  }

 private:
  template <typename... RemainingArgs>
  struct CustomKernelCallHelper;

  /* DeviceContext Helpers */

  // PD_SPECIALIZE_CustomKernelCallHelper_FOR_DEVICE_CONTEXT_C_API(PD_Context);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_DEVICE_CONTEXT(PD_Context);

  /* Input Helpers */

  // PD_SPECIALIZE_CustomKernelCallHelper_FOR_INPUT_C_API(PD_Tensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_INPUT(PD_Tensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_OPTIONAL_INPUT(PD_Tensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_INPUT(PD_Tensor);

  /* Attribute Helpers */

  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(bool);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(int32_t);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(int64_t);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(float);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(double);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(PD_DataType);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(PD_DataLayout);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(phi::custom_kernel::Place);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<bool>);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<int32_t>);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<int64_t>);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<float>);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<double>);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(std::string);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<std::string>);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      phi::custom_kernel::Scalar);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      phi::custom_kernel::IntArray);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<phi::custom_kernel::Scalar>);

  /* Output Helpers */

  // PD_SPECIALIZE_CustomKernelCallHelper_FOR_OUTPUT_C_API(PD_Tensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_OUTPUT(PD_Tensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_OUTPUT(PD_Tensor);

  /* End case */
  template <typename T>
  struct CustomKernelCallHelper<CustomTypeTag<T>> {
    template <int dev_ctx_idx, int in_idx, int attr_idx, int out_idx>
    static void Compute(PD_ExecutionContext *ctx,
                        DevCtx dev_ctx,
                        Args &... args) {
      static_assert(dev_ctx_idx > 0,
                    "Kernel should pass DeviceContext as argument.");
      static_assert(out_idx > 0, "Kernel should have output argument.");
      return kernel_fn(dev_ctx, args...);
    }
  };
};

template <typename Func>
struct CustomKernelArgsParseFunctor;

template <typename Return_, typename... Args_>
struct CustomKernelArgsParseFunctor<Return_ (*)(Args_...)> {
  using Args = std::tuple<Args_...>;
  enum : std::size_t { Arity = sizeof...(Args_) };
  using Indices = std::make_index_sequence<Arity>;
  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, Args>::type;

  CustomKernelArgsParseFunctor() {
    auto args_type = ParseArgType(Indices{});

    for (auto arg_type : args_type) {
      if (arg_type == std::type_index(typeid(const PD_Context *))) {
      } else if (arg_type == std::type_index(typeid(const PD_Tensor *)) ||
                 arg_type == std::type_index(typeid(
                                 const phi::custom_kernel::DenseTensor &))) {
        in_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_TENSOR);
      } else if (arg_type ==
                 std::type_index(
                     typeid(paddle::optional<
                            const phi::custom_kernel::DenseTensor &>))) {
        in_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_OPTIONAL_TENSOR);
      } else if (arg_type ==
                 std::type_index(typeid(
                     const std::vector<const phi::custom_kernel::DenseTensor *>
                         &))) {
        in_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_TENSOR);
      } else if (arg_type == std::type_index(typeid(bool))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_BOOL);
      } else if (arg_type == std::type_index(typeid(float))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_FLOAT32);
      } else if (arg_type == std::type_index(typeid(double))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_FLOAT64);
      } else if (arg_type == std::type_index(typeid(int32_t))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_INT32);
      } else if (arg_type == std::type_index(typeid(int64_t))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_INT64);
      } else if (arg_type ==
                 std::type_index(typeid(const phi::custom_kernel::Place &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_PLACE);
      } else if (arg_type == std::type_index(typeid(const std::string &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_STRING);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<bool> &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_BOOL);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<float> &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_FLOAT32);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<double> &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_FLOAT64);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<int32_t> &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_INT32);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<int64_t> &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_INT64);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<std::string> &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_STRING);
      } else if (arg_type ==
                 std::type_index(
                     typeid(const std::vector<phi::custom_kernel::Scalar> &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_SCALAR);
      } else if (arg_type ==
                 std::type_index(typeid(const phi::custom_kernel::Scalar &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_SCALAR);
      } else if (arg_type == std::type_index(typeid(
                                 const phi::custom_kernel::IntArray &))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_INT_ARRAY);
      } else if (arg_type == std::type_index(typeid(PD_DataType))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_DATA_TYPE);
      } else if (arg_type == std::type_index(typeid(PD_DataLayout))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_DATA_LAYOUT);
      } else if (arg_type == std::type_index(typeid(PD_DataLayout))) {
        attr_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_PLACE);
      } else if (arg_type == std::type_index(typeid(PD_Tensor *)) ||
                 arg_type == std::type_index(
                                 typeid(phi::custom_kernel::DenseTensor *))) {
        out_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_TENSOR);
      } else if (arg_type ==
                 std::type_index(
                     typeid(std::vector<phi::custom_kernel::DenseTensor *>))) {
        out_args_type.push_back(PD_ArgumentType::PD_ARG_TYPE_LIST_TENSOR);
      }
    }
  }

  std::vector<PD_ArgumentType> in_args_type;
  std::vector<PD_ArgumentType> attr_args_type;
  std::vector<PD_ArgumentType> out_args_type;

 private:
  template <std::size_t... INDEX>
  static std::vector<std::type_index> ParseArgType(
      std::index_sequence<INDEX...>) {
    return {std::type_index(typeid(Arg<INDEX>))...};
  }
};  // namespace phi

}  // namespace phi
