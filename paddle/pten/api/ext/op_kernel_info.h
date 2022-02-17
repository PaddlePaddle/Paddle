/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/pten/api/ext/op_meta_info.h"
#include "paddle/pten/core/kernel_registry.h"

/**
 * Custom Kernel Info Define.
 * Used to maintain custom kernel core information before registering.
 */

namespace paddle {

using DenseTensor = pten::DenseTensor;

//////////////// Op Kernel Info Map /////////////////
class PADDLE_API OpKernelInfoMap {
 public:
  static OpKernelInfoMap& Instance() {
    static OpKernelInfoMap g_custom_kernel_info_map;
    return g_custom_kernel_info_map;
  }

  pten::KernelNameMap& Kernels() { return kernels_; }
  const pten::KernelNameMap& GetMap() const;

 private:
  OpKernelInfoMap() = default;
  pten::KernelNameMap kernels_;

  PD_DISABLE_COPY_AND_ASSIGN(OpKernelInfoMap);
};

//////////////// Custom KernelRegistrar /////////////////
struct CustomKernelRegistrar {
 public:
  CustomKernelRegistrar(const char* kernel_name_cstr,
                        const char* backend_cstr,
                        pten::DataLayout layout,
                        pten::DataType dtype,
                        pten::KernelArgsParseFn args_parse_fn,
                        pten::KernelArgsDefFn args_def_fn,
                        pten::KernelFn kernel_fn,
                        void* variadic_kernel_fn) {
    ConstructCustomKernel(kernel_name_cstr,
                          backend_cstr,
                          layout,
                          dtype,
                          args_parse_fn,
                          args_def_fn,
                          kernel_fn,
                          variadic_kernel_fn);
  }

 private:
  pten::Backend GetOrRegisterBackend(const char* backend_cstr) {
    std::string backend(backend_cstr);
    if (backend == "CPU") {
      return pten::Backend::CPU;
    } else if (backend == "GPU") {
      return pten::Backend::GPU;
    } else if (backend == "XPU") {
      return pten::Backend::XPU;
    } else {
      auto device_type_id = pten::GetOrRegisterGlobalDeviceTypeId(backend);
      return static_cast<pten::Backend>(
          static_cast<size_t>(pten::Backend::NUM_BACKENDS) + device_type_id);
    }
  }
  void ConstructCustomKernel(const char* kernel_name_cstr,
                             const char* backend_cstr,
                             pten::DataLayout layout,
                             pten::DataType dtype,
                             pten::KernelArgsParseFn args_parse_fn,
                             pten::KernelArgsDefFn args_def_fn,
                             pten::KernelFn kernel_fn,
                             void* variadic_kernel_fn) {
    std::string kernel_name(kernel_name_cstr);
    pten::KernelKey kernel_key(
        GetOrRegisterBackend(backend_cstr), layout, dtype);
    pten::Kernel kernel(kernel_fn, variadic_kernel_fn);
    args_parse_fn(kernel_key, kernel.mutable_args_def());
    args_def_fn(kernel_key, &kernel);
    OpKernelInfoMap::Instance().Kernels()[kernel_name][kernel_key] = kernel;
  }
};
/////////////////////// Custom kernel register API /////////////////////////
// For inference: compile directly with framework
// Call after PD_REGISTER_KERNEL(...)
void RegisterAllCustomKernel();

// Using this api to load compiled custom kernel's dynamic library and
// register custom kernels
void LoadCustomKernelLib(const std::string& dso_name);

//////////////// Custom kernel register macro /////////////////////////////
// Refer to PT_REGISTER_KERNEL in paddle/pten/core/kernel_registry.h, we
// provide PD_REGISTER_KERNEL and PD_REGISTER_CUSTOM KERNEL which supports
// 2 template arguments.

#define PD_REGISTER_KERNEL(kernel_name, backend, layout, func, cpp_dtype, ...) \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                              \
      _reg_custom_kernel_ns_check_##kernel_name##_##backend##_##layout,        \
      "PD_REGISTER_KERNEL must be called in global namespace.");               \
  _PD_REGISTER_2TA_KERNEL(kernel_name,                                         \
                          backend,                                             \
                          ::pten::backend##Context,                            \
                          layout,                                              \
                          func,                                                \
                          cpp_dtype,                                           \
                          ##__VA_ARGS__)

#define PD_REGISTER_CUSTOM_KERNEL(                                      \
    kernel_name, backend, layout, func, cpp_dtype, ...)                 \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                       \
      _reg_custom_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_REGISTER_KERNEL must be called in global namespace.");        \
  _PD_REGISTER_2TA_KERNEL(kernel_name,                                  \
                          backend,                                      \
                          ::pten::CustomContext,                        \
                          layout,                                       \
                          func,                                         \
                          cpp_dtype,                                    \
                          ##__VA_ARGS__)

// WIN32 is not supported
#define _PD_REGISTER_2TA_KERNEL(                                            \
    kernel_name, backend, context, layout, meta_kernel_fn, cpp_dtype, ...)  \
  PD_KERNEL_INSTANTIATION(                                                  \
      meta_kernel_fn, backend, context, cpp_dtype, ##__VA_ARGS__);          \
  static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
      const ::pten::KernelKey& kernel_key, ::pten::Kernel* kernel);         \
  PD_KERNEL_REGISTRAR_INIT(                                                 \
      kernel_name,                                                          \
      backend,                                                              \
      context,                                                              \
      layout,                                                               \
      &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,        \
      meta_kernel_fn,                                                       \
      cpp_dtype,                                                            \
      ##__VA_ARGS__);                                                       \
  void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
      const ::pten::KernelKey& kernel_key, ::pten::Kernel* kernel)

#define PD_KERNEL_INSTANTIATION(                               \
    meta_kernel_fn, backend, context, cpp_dtype, ...)          \
  _PD_KERNEL_INSTANTIATION(PT_NARGS(cpp_dtype, ##__VA_ARGS__), \
                           meta_kernel_fn,                     \
                           backend,                            \
                           context,                            \
                           cpp_dtype,                          \
                           ##__VA_ARGS__)

#define _PD_KERNEL_INSTANTIATION(                        \
    N, meta_kernel_fn, backend, context, cpp_dtype, ...) \
  PT_CONCATENATE(_PD_KERNEL_INSTANTIATION_, N)           \
  (meta_kernel_fn, backend, context, cpp_dtype, ##__VA_ARGS__)

#define _PD_KERNEL_INSTANTIATION_1(                   \
    meta_kernel_fn, backend, context, cpp_dtype, ...) \
  template decltype(                                  \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>
#define _PD_KERNEL_INSTANTIATION_2(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_1(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_3(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_2(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_4(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_3(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_5(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_4(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_6(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_5(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_7(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_6(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_8(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_7(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_9(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_8(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_10(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_9(                                       \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_11(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_10(                                      \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_12(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_11(                                      \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_13(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_12(                                      \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_14(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_13(                                      \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_15(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PD_KERNEL_INSTANTIATION_14(                                      \
      meta_kernel_fn, backend, context, ##__VA_ARGS__))

#define PD_KERNEL_REGISTRAR_INIT(kernel_name,                   \
                                 backend,                       \
                                 context,                       \
                                 layout,                        \
                                 args_def_fn,                   \
                                 meta_kernel_fn,                \
                                 cpp_dtype,                     \
                                 ...)                           \
  _PD_KERNEL_REGISTRAR_INIT(PT_NARGS(cpp_dtype, ##__VA_ARGS__), \
                            kernel_name,                        \
                            backend,                            \
                            context,                            \
                            layout,                             \
                            args_def_fn,                        \
                            meta_kernel_fn,                     \
                            cpp_dtype,                          \
                            ##__VA_ARGS__)

// clang-format off

/* The =pre-commit always treats this macro into the wrong format,
  and multi-line macros cannot be skipped with NOLINT.*/
#define _PD_KERNEL_REGISTRAR_INIT(N,              \
                                  kernel_name,    \
                                  backend,        \
                                  context,        \
                                  layout,         \
                                  args_def_fn,    \
                                  meta_kernel_fn, \
                                  cpp_dtype,      \
                                  ...)            \
  PT_CONCATENATE(_PD_KERNEL_REGISTRAR_INIT_, N) ( \
    kernel_name,                                  \
    backend,                                      \
    context,                                      \
    layout,                                       \
    PT_ID,                                        \
    args_def_fn,                                  \
    meta_kernel_fn,                               \
    cpp_dtype,                                    \
    ##__VA_ARGS__)
// clang-format on

#define _PD_KERNEL_REGISTRAR_INIT_1(kernel_name,                          \
                                    backend,                              \
                                    context,                              \
                                    layout,                               \
                                    registrar_id,                         \
                                    args_def_fn,                          \
                                    meta_kernel_fn,                       \
                                    cpp_dtype,                            \
                                    ...)                                  \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(            \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,        \
      registrar_id)(                                                      \
      #kernel_name,                                                       \
      #backend,                                                           \
      DATALAYOUT(layout),                                              \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),       \
      ::pten::KernelArgsParseFunctor<decltype(                            \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                   \
      args_def_fn,                                                        \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                      \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));            \
  int TouchCustomKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
    return 0;                                                             \
  }
#define _PD_KERNEL_REGISTRAR_INIT_2(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_1(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_3(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_2(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_4(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_3(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_5(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_4(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_6(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_5(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_7(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_6(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_8(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_7(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_9(kernel_name,                    \
                                    backend,                        \
                                    context,                        \
                                    layout,                         \
                                    registrar_id,                   \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_8(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_10(kernel_name,                   \
                                     backend,                       \
                                     context,                       \
                                     layout,                        \
                                     registrar_id,                  \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_9(kernel_name,                \
                                        backend,                    \
                                        context,                    \
                                        layout,                     \
                                        PT_ID,                      \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_11(kernel_name,                   \
                                     backend,                       \
                                     context,                       \
                                     layout,                        \
                                     registrar_id,                  \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_10(kernel_name,               \
                                         backend,                   \
                                         context,                   \
                                         layout,                    \
                                         PT_ID,                     \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_12(kernel_name,                   \
                                     backend,                       \
                                     context,                       \
                                     layout,                        \
                                     registrar_id,                  \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_11(kernel_name,               \
                                         backend,                   \
                                         context,                   \
                                         layout,                    \
                                         PT_ID,                     \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_13(kernel_name,                   \
                                     backend,                       \
                                     context,                       \
                                     layout,                        \
                                     registrar_id,                  \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_12(kernel_name,               \
                                         backend,                   \
                                         context,                   \
                                         layout,                    \
                                         PT_ID,                     \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_14(kernel_name,                   \
                                     backend,                       \
                                     context,                       \
                                     layout,                        \
                                     registrar_id,                  \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_13(kernel_name,               \
                                         backend,                   \
                                         context,                   \
                                         layout,                    \
                                         PT_ID,                     \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_15(kernel_name,                   \
                                     backend,                       \
                                     context,                       \
                                     layout,                        \
                                     registrar_id,                  \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::paddle::CustomKernelRegistrar PT_CONCATENATE(      \
      __reg_custom_kernel_##kernel_name##_##backend##_##layout##_,  \
      registrar_id)(                                                \
      #kernel_name,                                                 \
      #backend,                                                     \
      DATALAYOUT(layout),                                        \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,             \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));      \
  PT_EXPAND(_PD_KERNEL_REGISTRAR_INIT_14(kernel_name,               \
                                         backend,                   \
                                         context,                   \
                                         layout,                    \
                                         PT_ID,                     \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         ##__VA_ARGS__))

}  // namespace paddle
