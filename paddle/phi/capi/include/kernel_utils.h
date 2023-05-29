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

#include "paddle/phi/capi/include/common.h"

#if !defined(_WIN32)

namespace phi {
namespace capi {

#define CUSTOM_PHI_KERNEL(...) \
  ::phi::capi::CustomKernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define CUSTOM_PHI_VARIADIC_KERNEL(...)                      \
  reinterpret_cast<void *>(                                  \
      &::phi::capi::CustomKernelImpl<decltype(&__VA_ARGS__), \
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

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_1(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype)                           \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  int TouchCustomKernelSymbolFor_##kernel_name##_##backend##_##layout() {      \
    return 0;                                                                  \
  }

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_2(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_1(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_3(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_2(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_4(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_3(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_5(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_4(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_6(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_5(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_7(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_6(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_8(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_7(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_9(registrar_class,                     \
                                          kernel_name,                         \
                                          backend,                             \
                                          layout,                              \
                                          registrar_id,                        \
                                          meta_kernel_fn,                      \
                                          cpp_dtype,                           \
                                          ...)                                 \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_8(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_10(registrar_class,                    \
                                           kernel_name,                        \
                                           backend,                            \
                                           layout,                             \
                                           registrar_id,                       \
                                           meta_kernel_fn,                     \
                                           cpp_dtype,                          \
                                           ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_9(registrar_class,                       \
                                        kernel_name,                           \
                                        backend,                               \
                                        layout,                                \
                                        PD_CUSTOM_PHI_KERNEL_ID,               \
                                        meta_kernel_fn,                        \
                                        __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_11(registrar_class,                    \
                                           kernel_name,                        \
                                           backend,                            \
                                           layout,                             \
                                           registrar_id,                       \
                                           meta_kernel_fn,                     \
                                           cpp_dtype,                          \
                                           ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_10(registrar_class,                      \
                                         kernel_name,                          \
                                         backend,                              \
                                         layout,                               \
                                         PD_CUSTOM_PHI_KERNEL_ID,              \
                                         meta_kernel_fn,                       \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_12(registrar_class,                    \
                                           kernel_name,                        \
                                           backend,                            \
                                           layout,                             \
                                           registrar_id,                       \
                                           meta_kernel_fn,                     \
                                           cpp_dtype,                          \
                                           ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_11(registrar_class,                      \
                                         kernel_name,                          \
                                         backend,                              \
                                         layout,                               \
                                         PD_CUSTOM_PHI_KERNEL_ID,              \
                                         meta_kernel_fn,                       \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_13(registrar_class,                    \
                                           kernel_name,                        \
                                           backend,                            \
                                           layout,                             \
                                           registrar_id,                       \
                                           meta_kernel_fn,                     \
                                           cpp_dtype,                          \
                                           ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_12(registrar_class,                      \
                                         kernel_name,                          \
                                         backend,                              \
                                         layout,                               \
                                         PD_CUSTOM_PHI_KERNEL_ID,              \
                                         meta_kernel_fn,                       \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_14(registrar_class,                    \
                                           kernel_name,                        \
                                           backend,                            \
                                           layout,                             \
                                           registrar_id,                       \
                                           meta_kernel_fn,                     \
                                           cpp_dtype,                          \
                                           ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_13(registrar_class,                      \
                                         kernel_name,                          \
                                         backend,                              \
                                         layout,                               \
                                         PD_CUSTOM_PHI_KERNEL_ID,              \
                                         meta_kernel_fn,                       \
                                         __VA_ARGS__))

#define _PD_BUILD_KERNEL_REGISTRAR_INIT_15(registrar_class,                    \
                                           kernel_name,                        \
                                           backend,                            \
                                           layout,                             \
                                           registrar_id,                       \
                                           meta_kernel_fn,                     \
                                           cpp_dtype,                          \
                                           ...)                                \
  static const registrar_class<cpp_dtype> PD_CUSTOM_PHI_KERNEL_CONCATENATE(    \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id); \
  PD_CUSTOM_PHI_KERNEL_EXPAND(                                                 \
      _PD_BUILD_KERNEL_REGISTRAR_INIT_14(registrar_class,                      \
                                         kernel_name,                          \
                                         backend,                              \
                                         layout,                               \
                                         PD_CUSTOM_PHI_KERNEL_ID,              \
                                         meta_kernel_fn,                       \
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

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_DEVICE_CONTEXT(dev_ctx)      \
  template <typename... Tail>                                                 \
  struct CustomKernelCallHelper<const dev_ctx &, Tail...> {                   \
    template <int dev_ctx_idx,                                                \
              int in_idx,                                                     \
              int attr_idx,                                                   \
              int out_idx,                                                    \
              typename... PreviousArgs>                                       \
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {      \
      static_assert(in_idx == 0,                                              \
                    "Kernel's DeviceContext should appear before Inputs.");   \
      static_assert(                                                          \
          attr_idx == 0,                                                      \
          "Kernel's DeviceContext should appear before Attributes.");         \
      static_assert(out_idx == 0,                                             \
                    "Kernel's DeviceContext should appear before Outputs.");  \
      dev_ctx arg = PD_GetDeviceContext(ctx);                                 \
      CustomKernelCallHelper<Tail...>::                                       \
          template Compute<dev_ctx_idx + 1, in_idx, attr_idx, out_idx>(       \
              ctx, pargs..., arg);                                            \
    }                                                                         \
    template <int idx, typename... PreviousArgs>                              \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,     \
                                PreviousArgs &...pargs) {                     \
      const dev_ctx &arg = std::get<idx>(ctx);                                \
      auto dev_ctx_wrapper = phi::capi::DeviceContext(                        \
          reinterpret_cast<PD_DeviceContext *>(const_cast<dev_ctx *>(&arg))); \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx +  \
                                                                       1>(    \
          ctx, pargs..., dev_ctx_wrapper);                                    \
    }                                                                         \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_INPUT(tensor_type)          \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<const tensor_type &, Tail...> {              \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {     \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      const tensor_type arg = PD_InputAt(ctx, in_idx);                       \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
    template <int idx, typename... PreviousArgs>                             \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,    \
                                PreviousArgs &...pargs) {                    \
      const tensor_type &arg = std::get<idx>(ctx);                           \
      auto tensor = phi::capi::DenseTensor(                                  \
          reinterpret_cast<PD_Tensor *>(const_cast<tensor_type *>(&arg)));   \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx + \
                                                                       1>(   \
          ctx, pargs..., tensor);                                            \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_OPTIONAL_INPUT(tensor_type) \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<const paddle::optional<tensor_type> &,       \
                                Tail...> {                                   \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {     \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      auto arg = PD_OptionalInputAt(ctx, in_idx);                            \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
    template <int idx, typename... PreviousArgs>                             \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,    \
                                PreviousArgs &...pargs) {                    \
      auto &arg = std::get<idx>(ctx);                                        \
      paddle::optional<tensor_type> tensor =                                 \
          PD_OptionalTensor(reinterpret_cast<PD_Tensor *>(                   \
              const_cast<paddle::optional<tensor_type> *>(&arg)));           \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx + \
                                                                       1>(   \
          ctx, pargs..., tensor);                                            \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_INPUT(tensor_type)    \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<const std::vector<const tensor_type *> &,    \
                                Tail...> {                                   \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {     \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      auto arg = PD_MultiInputAt(ctx, in_idx);                               \
      std::vector<const tensor_type *> tensor_ptr_vec;                       \
      for (auto &tensor : arg) {                                             \
        tensor_ptr_vec.push_back(tensor.raw_data() ? &tensor : nullptr);     \
      }                                                                      \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(      \
              ctx, pargs..., tensor_ptr_vec);                                \
    }                                                                        \
    template <int idx, typename... PreviousArgs>                             \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,    \
                                PreviousArgs &...pargs) {                    \
      auto &arg = std::get<idx>(ctx);                                        \
      auto tensor_vec = PD_TensorVector(reinterpret_cast<PD_Tensor *>(       \
          const_cast<std::vector<const tensor_type *> *>(&arg)));            \
      std::vector<const tensor_type *> tensor_ptr_vec;                       \
      for (auto &tensor : tensor_vec) {                                      \
        tensor_ptr_vec.push_back(tensor.raw_data() ? &tensor : nullptr);     \
      }                                                                      \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx + \
                                                                       1>(   \
          ctx, pargs..., tensor_ptr_vec);                                    \
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
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {     \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Attributes should appear before Outputs.");    \
      attr_type arg = PD_AttrAt<attr_type>(ctx, attr_idx);                   \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx + 1, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
    template <int idx, typename... PreviousArgs>                             \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,    \
                                PreviousArgs &...pargs) {                    \
      auto &arg = std::get<idx>(ctx);                                        \
      auto attr = PD_Attr<attr_type>(reinterpret_cast<void *>(&arg));        \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx + \
                                                                       1>(   \
          ctx, pargs..., attr);                                              \
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
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {     \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Attributes should appear before Outputs.");    \
      attr_type arg = PD_AttrAt<attr_type>(ctx, attr_idx);                   \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx + 1, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
    template <int idx, typename... PreviousArgs>                             \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,    \
                                PreviousArgs &...pargs) {                    \
      const attr_type &arg = std::get<idx>(ctx);                             \
      auto attr = PD_Attr<attr_type>(                                        \
          reinterpret_cast<void *>(const_cast<attr_type *>(&arg)));          \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx + \
                                                                       1>(   \
          ctx, pargs..., attr);                                              \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_OUTPUT(tensor_type)         \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<tensor_type *, Tail...> {                    \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {     \
      auto arg = PD_OutputAt(ctx, out_idx);                                  \
      tensor_type *ptr = (arg.raw_data() ? &arg : nullptr);                  \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx, out_idx + 1>(      \
              ctx, pargs..., ptr);                                           \
    }                                                                        \
    template <int idx, typename... PreviousArgs>                             \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,    \
                                PreviousArgs &...pargs) {                    \
      tensor_type *arg = std::get<idx>(ctx);                                 \
      auto tensor =                                                          \
          phi::capi::DenseTensor(reinterpret_cast<PD_Tensor *>(arg));        \
      auto tensor_ptr = tensor.raw_data() ? &tensor : nullptr;               \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx + \
                                                                       1>(   \
          ctx, pargs..., tensor_ptr);                                        \
    }                                                                        \
  }

#define PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_OUTPUT(tensor_type)   \
  template <typename... Tail>                                                \
  struct CustomKernelCallHelper<std::vector<tensor_type *>, Tail...> {       \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(PD_KernelContext *ctx, PreviousArgs &...pargs) {     \
      auto arg = PD_MultiOutputAt(ctx, out_idx);                             \
      std::vector<tensor_type *> tensor_ptr_vec;                             \
      for (auto &tensor : arg) {                                             \
        tensor_ptr_vec.push_back(tensor.raw_data() ? &tensor : nullptr);     \
      }                                                                      \
      CustomKernelCallHelper<Tail...>::                                      \
          template Compute<dev_ctx_idx, in_idx, attr_idx, out_idx + 1>(      \
              ctx, pargs..., tensor_ptr_vec);                                \
    }                                                                        \
    template <int idx, typename... PreviousArgs>                             \
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,    \
                                PreviousArgs &...pargs) {                    \
      std::vector<tensor_type *> &arg = std::get<idx>(ctx);                  \
      auto tensor_vec = PD_TensorVector(reinterpret_cast<PD_Tensor *>(       \
          const_cast<std::vector<tensor_type *> *>(&arg)));                  \
      std::vector<tensor_type *> tensor_ptr_vec;                             \
      for (auto &tensor : tensor_vec) {                                      \
        tensor_ptr_vec.push_back(tensor.raw_data() ? &tensor : nullptr);     \
      }                                                                      \
      return CustomKernelCallHelper<Tail...>::template VariadicCompute<idx + \
                                                                       1>(   \
          ctx, pargs..., tensor_ptr_vec);                                    \
    }                                                                        \
  }

template <typename T>
struct CustomTypeTag {};

template <typename Fn, Fn fn>
struct CustomKernelImpl;

template <typename Return,
          typename DevCtx,
          typename... Args,
          Return (*kernel_fn)(DevCtx, Args...)>
struct CustomKernelImpl<Return (*)(DevCtx, Args...), kernel_fn> {
  static void Compute(PD_KernelContext *ctx) {
    CustomKernelCallHelper<DevCtx, Args..., CustomTypeTag<int>>::
        template Compute<0, 0, 0, 0>(ctx);
  }

  static void VariadicCompute(DevCtx dev_ctx, Args... args) {
    const std::tuple<DevCtx, Args &...> args_tuple(dev_ctx, args...);
    return CustomKernelCallHelper<DevCtx, Args..., CustomTypeTag<int>>::
        template VariadicCompute<0>(args_tuple);
  }

 private:
  template <typename... RemainingArgs>
  struct CustomKernelCallHelper;

  /* DeviceContext Helpers */

  PD_SPECIALIZE_CustomKernelCallHelper_FOR_DEVICE_CONTEXT(
      phi::capi::DeviceContext);

  /* Input Helpers */

  PD_SPECIALIZE_CustomKernelCallHelper_FOR_INPUT(phi::capi::DenseTensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_OPTIONAL_INPUT(
      phi::capi::DenseTensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_INPUT(phi::capi::DenseTensor);

  /* Attribute Helpers */

  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(bool);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(int32_t);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(int64_t);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(float);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(double);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(PD_DataType);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(PD_DataLayout);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_ATTRIBUTE(phi::capi::Place);
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
      phi::capi::Scalar);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      phi::capi::IntArray);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_CONST_ATTRIBUTE_REF(
      std::vector<phi::capi::Scalar>);

  /* Output Helpers */

  PD_SPECIALIZE_CustomKernelCallHelper_FOR_OUTPUT(phi::capi::DenseTensor);
  PD_SPECIALIZE_CustomKernelCallHelper_FOR_MULTI_OUTPUT(phi::capi::DenseTensor);

  /* End case */
  template <typename T>
  struct CustomKernelCallHelper<CustomTypeTag<T>> {
    template <int dev_ctx_idx, int in_idx, int attr_idx, int out_idx>
    static void Compute(PD_KernelContext *ctx, DevCtx dev_ctx, Args &...args) {
      static_assert(dev_ctx_idx > 0,
                    "Kernel should pass DeviceContext as argument.");
      static_assert(out_idx > 0, "Kernel should have output argument.");
      return kernel_fn(dev_ctx, args...);
    }

    template <int idx>
    static void VariadicCompute(const std::tuple<DevCtx, Args &...> &ctx,
                                DevCtx dev_ctx,
                                Args... args) {
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
      if (arg_type ==
          std::type_index(typeid(const phi::capi::DeviceContext *))) {
      } else if (arg_type ==
                 std::type_index(typeid(const phi::capi::DenseTensor &))) {
        in_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_TENSOR);
      } else if (arg_type ==
                 std::type_index(typeid(
                     const paddle::optional<phi::capi::DenseTensor> &))) {
        in_args_type.push_back(
            PD_KernelArgumentType::PD_ARG_TYPE_OPTIONAL_TENSOR);
      } else if (arg_type ==
                 std::type_index(typeid(
                     const std::vector<const phi::capi::DenseTensor *> &))) {
        in_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_LIST_TENSOR);
      } else if (arg_type ==
                 std::type_index(
                     typeid(const paddle::optional<
                            std::vector<const phi::capi::DenseTensor *>> &))) {
        in_args_type.push_back(
            PD_KernelArgumentType::PD_ARG_TYPE_OPTIONAL_MULTI_TENSOR);
      } else if (arg_type == std::type_index(typeid(bool))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_BOOL);
      } else if (arg_type == std::type_index(typeid(float))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_FLOAT32);
      } else if (arg_type == std::type_index(typeid(double))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_FLOAT64);
      } else if (arg_type == std::type_index(typeid(int32_t))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_INT32);
      } else if (arg_type == std::type_index(typeid(int64_t))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_INT64);
      } else if (arg_type ==
                 std::type_index(typeid(const phi::capi::Place &))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_PLACE);
      } else if (arg_type == std::type_index(typeid(const std::string &))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_STRING);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<bool> &))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_LIST_BOOL);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<float> &))) {
        attr_args_type.push_back(
            PD_KernelArgumentType::PD_ARG_TYPE_LIST_FLOAT32);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<double> &))) {
        attr_args_type.push_back(
            PD_KernelArgumentType::PD_ARG_TYPE_LIST_FLOAT64);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<int32_t> &))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_LIST_INT32);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<int64_t> &))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_LIST_INT64);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<std::string> &))) {
        attr_args_type.push_back(
            PD_KernelArgumentType::PD_ARG_TYPE_LIST_STRING);
      } else if (arg_type == std::type_index(typeid(
                                 const std::vector<phi::capi::Scalar> &))) {
        attr_args_type.push_back(
            PD_KernelArgumentType::PD_ARG_TYPE_LIST_SCALAR);
      } else if (arg_type ==
                 std::type_index(typeid(const phi::capi::Scalar &))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_SCALAR);
      } else if (arg_type ==
                 std::type_index(typeid(const phi::capi::IntArray &))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_INT_ARRAY);
      } else if (arg_type == std::type_index(typeid(PD_DataType))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_DATA_TYPE);
      } else if (arg_type == std::type_index(typeid(PD_DataLayout))) {
        attr_args_type.push_back(
            PD_KernelArgumentType::PD_ARG_TYPE_DATA_LAYOUT);
      } else if (arg_type == std::type_index(typeid(PD_DataLayout))) {
        attr_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_PLACE);
      } else if (arg_type ==
                 std::type_index(typeid(phi::capi::DenseTensor *))) {
        out_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_TENSOR);
      } else if (arg_type == std::type_index(typeid(
                                 std::vector<phi::capi::DenseTensor *>))) {
        out_args_type.push_back(PD_KernelArgumentType::PD_ARG_TYPE_LIST_TENSOR);
      }
    }
  }

  std::vector<PD_KernelArgumentType> in_args_type;
  std::vector<PD_KernelArgumentType> attr_args_type;
  std::vector<PD_KernelArgumentType> out_args_type;

 private:
  template <std::size_t... INDEX>
  static std::vector<std::type_index> ParseArgType(
      std::index_sequence<INDEX...>) {
    return {std::type_index(typeid(Arg<INDEX>))...};
  }
};

}  // namespace capi
}  // namespace phi

#endif
