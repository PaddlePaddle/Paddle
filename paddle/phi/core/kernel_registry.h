//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "paddle/phi/core/custom_kernel.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/phi/core/enforce.h"

namespace phi {

#define BACKEND(arg__) phi::Backend::arg__
#define DATALAYOUT(arg__) phi::DataLayout::arg__
#define DATATYPE(arg__) phi::DataType::arg__

template <typename Func>
struct KernelArgsParseFunctor;

template <typename Return_, typename... Args_>
struct KernelArgsParseFunctor<Return_ (*)(Args_...)> {
  using Args = std::tuple<Args_...>;
  enum : std::size_t { Arity = sizeof...(Args_) };
  using Indices = std::make_index_sequence<Arity>;
  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, Args>::type;

  static void Parse(const KernelKey& default_key, KernelArgsDef* args_def) {
    // TODO(chenweihang): The fluid Tensor's default layout is NCHW,
    // it is not same as kernel's layout, we should fix this error on
    // fluid Tensor
    auto default_tensor_layout = phi::DataLayout::NCHW;
    if (default_key.layout() != phi::DataLayout::ANY) {
      default_tensor_layout = default_key.layout();
    }
    auto args_type = ParseArgType(Indices{});
    for (auto arg_type : args_type) {
      if (arg_type == std::type_index(typeid(const CPUContext&))
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
          ||
          arg_type == std::type_index(typeid(const GPUContext&))) {
#elif defined(PADDLE_WITH_XPU)
          ||
          arg_type == std::type_index(typeid(const XPUContext&))) {
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
          ||
          arg_type == std::type_index(typeid(const CustomContext&))) {
#else
              ) {
#endif
        // do nothing, skip context arg now
      } else if (arg_type == std::type_index(typeid(const DenseTensor&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 paddle::optional<const DenseTensor&>))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<DenseTensor>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(const SelectedRows&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(DenseTensor*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type ==
                 std::type_index(typeid(std::vector<DenseTensor*>))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(SelectedRows*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else {
        // Attribute deal with
        // TODO(chenweihang): now here allow any types of attribute, maybe
        // should add limits here
        args_def->AppendAttribute(arg_type);
      }
    }
  }

 private:
  template <std::size_t... INDEX>
  static std::vector<std::type_index> ParseArgType(
      std::index_sequence<INDEX...>) {
    return {std::type_index(typeid(Arg<INDEX>))...};
  }
};

// NOTE: used for making a difference between inner or outer registration.
enum class RegType : uint8_t {
  INNER = 0,
  OUTER,
};

// TODO(chenweihang): Polish the kernel selection logic, support the selection
// of ALL_DTYPE kernel, and simplify the constructor
struct KernelRegistrar {
 public:
  KernelRegistrar(RegType reg_type,
                  const char* kernel_name_cstr,
                  const char* backend_cstr,
                  DataLayout layout,
                  DataType dtype,
                  KernelArgsParseFn args_parse_fn,
                  KernelArgsDefFn args_def_fn,
                  KernelFn kernel_fn,
                  void* variadic_kernel_fn) {
    ConstructKernel(reg_type,
                    kernel_name_cstr,
                    backend_cstr,
                    layout,
                    dtype,
                    args_parse_fn,
                    args_def_fn,
                    kernel_fn,
                    variadic_kernel_fn);
  }

  KernelRegistrar(RegType reg_type,
                  const char* kernel_name_cstr,
                  const char* backend_cstr,
                  DataLayout layout,
                  KernelArgsParseFn args_parse_fn,
                  KernelArgsDefFn args_def_fn,
                  KernelFn kernel_fn,
                  void* variadic_kernel_fn) {
    for (size_t dtype = static_cast<size_t>(DataType::BOOL);
         dtype != static_cast<size_t>(DataType::NUM_DATA_TYPES);
         dtype++) {
      // NOTE(zhiqiu): why skip these types, because fluid kernel has no kernel
      // of these type.
      if (dtype == static_cast<size_t>(DataType::UINT32) ||
          dtype == static_cast<size_t>(DataType::UINT64) ||
          dtype == static_cast<size_t>(DataType::UINT16)) {
        continue;
      }
      ConstructKernel(reg_type,
                      kernel_name_cstr,
                      backend_cstr,
                      layout,
                      static_cast<DataType>(dtype),
                      args_parse_fn,
                      args_def_fn,
                      kernel_fn,
                      variadic_kernel_fn);
    }
  }

 private:
  void ConstructKernel(RegType reg_type,
                       const char* kernel_name_cstr,
                       const char* backend_cstr,
                       DataLayout layout,
                       DataType dtype,
                       KernelArgsParseFn args_parse_fn,
                       KernelArgsDefFn args_def_fn,
                       KernelFn kernel_fn,
                       void* variadic_kernel_fn) {
    std::string kernel_name(kernel_name_cstr);
    KernelKey kernel_key(
        paddle::experimental::StringToBackend(backend_cstr), layout, dtype);
    Kernel kernel(kernel_fn, variadic_kernel_fn);
    args_parse_fn(kernel_key, kernel.mutable_args_def());
    args_def_fn(kernel_key, &kernel);
    if (reg_type == RegType::INNER) {
      KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
    } else {
      CustomKernelMap::Instance().Kernels()[kernel_name][kernel_key] = kernel;
    }
  }
};

/**
 * Reference:
 *
 *   https://stackoverflow.com/questions/1872220/is-it-possible-to-iterate-over-arguments-in-variadic-macros
 *   https://stackoverflow.com/questions/9183993/msvc-variadic-macro-expansion?rq=1
 *   https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
 *
 * Very carefully tiptoeing around an MSVC bug where it improperly expands
 * __VA_ARGS__ as a single token in argument lists.  See these URLs for details:
 *
 *   http://connect.microsoft.com/VisualStudio/feedback/details/380090/variadic-macro-replacement
 *   http://cplusplus.co.il/2010/07/17/variadic-macro-to-count-number-of-arguments/#comment-644
 */
#define PT_NARGS(...) _PT_NARGS((__VA_ARGS__, _PT_RESQ_N()))
#define _PT_NARGS(...) _PT_ARG_N(__VA_ARGS__)
#define _PT_ARG_N_EXPAND(                                                     \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) \
  N
#define _PT_ARG_N(args) _PT_ARG_N_EXPAND args
#define _PT_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

/** PD_REGISTER_KERNEL
 *
 * The most frequently used kernel registration macro, used for kernel
 * registration with only data type as template parameter, and the function
 * pointer of the corresponding data type is automatically instantiated
 * during registration.
 *
 * Note: `2TA` means `2 template argument`
 */
#define PD_REGISTER_KERNEL(kernel_name, backend, layout, meta_kernel_fn, ...) \
  _PD_REGISTER_KERNEL(::phi::RegType::INNER,                                  \
                      kernel_name,                                            \
                      backend,                                                \
                      ::phi::backend##Context,                                \
                      layout,                                                 \
                      meta_kernel_fn,                                         \
                      __VA_ARGS__)

#define _PD_REGISTER_KERNEL(                                               \
    reg_type, kernel_name, backend, context, layout, meta_kernel_fn, ...)  \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                       \
      PD_REGISTER_tp_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_REGISTER_KERNEL must be called in global namespace.");           \
  PT_EXPAND(_PD_REGISTER_2TA_KERNEL(reg_type,                              \
                                    kernel_name,                           \
                                    backend,                               \
                                    context,                               \
                                    layout,                                \
                                    meta_kernel_fn,                        \
                                    __VA_ARGS__))

#ifndef _WIN32
#define _PD_REGISTER_2TA_KERNEL(                                            \
    reg_type, kernel_name, backend, context, layout, meta_kernel_fn, ...)   \
  PT_KERNEL_INSTANTIATION(meta_kernel_fn, backend, context, __VA_ARGS__);   \
  static void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
  PT_KERNEL_REGISTRAR_INIT(                                                 \
      reg_type,                                                             \
      kernel_name,                                                          \
      backend,                                                              \
      context,                                                              \
      layout,                                                               \
      &__PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,        \
      meta_kernel_fn,                                                       \
      __VA_ARGS__);                                                         \
  void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
#else
/**
 * `template decltype(fn) fn` can work on gcc and clang,
 * but msvc will failed, error like:
 *
 *   error C2206: typedef cannot be used for function definition
 *
 * reference:
 *
 *   https://stackoverflow.com/questions/63989585/explicit-instantiation-of-function-using-decltype-work-on-g-but-not-on-visua
 *
 * And msvc can work without template instantiation
 */
#define _PD_REGISTER_2TA_KERNEL(                                            \
    reg_type, kernel_name, backend, context, layout, meta_kernel_fn, ...)   \
  static void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
  PT_EXPAND(PT_KERNEL_REGISTRAR_INIT(                                       \
      reg_type,                                                             \
      kernel_name,                                                          \
      backend,                                                              \
      context,                                                              \
      layout,                                                               \
      &__PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,        \
      meta_kernel_fn,                                                       \
      __VA_ARGS__));                                                        \
  void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
#endif

#define PT_KERNEL_INSTANTIATION(meta_kernel_fn, backend, context, ...) \
  _PT_KERNEL_INSTANTIATION(                                            \
      PT_NARGS(__VA_ARGS__), meta_kernel_fn, backend, context, __VA_ARGS__)

#define _PT_KERNEL_INSTANTIATION(N, meta_kernel_fn, backend, context, ...) \
  PT_CONCATENATE(_PT_KERNEL_INSTANTIATION_, N)                             \
  (meta_kernel_fn, backend, context, __VA_ARGS__)

#define _PT_KERNEL_INSTANTIATION_1(              \
    meta_kernel_fn, backend, context, cpp_dtype) \
  template decltype(                             \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>
#define _PT_KERNEL_INSTANTIATION_2(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_1(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_3(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_2(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_4(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_3(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_5(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_4(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_6(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_5(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_7(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_6(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_8(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_7(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_9(                                           \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_8(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_10(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_9(                                       \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_11(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_10(                                      \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_12(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_11(                                      \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_13(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_12(                                      \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_14(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_13(                                      \
      meta_kernel_fn, backend, context, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_15(                                          \
    meta_kernel_fn, backend, context, cpp_dtype, ...)                         \
  template decltype(                                                          \
      meta_kernel_fn<cpp_dtype, context>) meta_kernel_fn<cpp_dtype, context>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_14(                                      \
      meta_kernel_fn, backend, context, __VA_ARGS__))

#define PT_KERNEL_REGISTRAR_INIT(reg_type,                   \
                                 kernel_name,                \
                                 backend,                    \
                                 context,                    \
                                 layout,                     \
                                 args_def_fn,                \
                                 meta_kernel_fn,             \
                                 ...)                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT(PT_NARGS(__VA_ARGS__), \
                                      reg_type,              \
                                      kernel_name,           \
                                      backend,               \
                                      context,               \
                                      layout,                \
                                      args_def_fn,           \
                                      meta_kernel_fn,        \
                                      __VA_ARGS__))

// clang-format off

/* The =pre-commit always treats this macro into the wrong format,
  and multi-line macros cannot be skipped with NOLINT.*/
#define _PT_KERNEL_REGISTRAR_INIT(N,                       \
                                  reg_type,                \
                                  kernel_name,             \
                                  backend,                 \
                                  context,                 \
                                  layout,                  \
                                  args_def_fn,             \
                                  meta_kernel_fn,          \
                                  ...)                     \
  PT_EXPAND(PT_CONCATENATE(_PT_KERNEL_REGISTRAR_INIT_, N) ( \
    reg_type,                                              \
    kernel_name,                                           \
    backend,                                               \
    context,                                               \
    layout,                                                \
    PT_ID,                                                 \
    args_def_fn,                                           \
    meta_kernel_fn,                                        \
    __VA_ARGS__))

// clang-format on

#define _PT_KERNEL_REGISTRAR_INIT_1(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype)                                \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { return 0; }
#define _PT_KERNEL_REGISTRAR_INIT_2(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_1(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_3(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_2(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_4(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_3(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_5(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_4(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_6(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_5(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_7(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_6(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_8(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_7(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_9(reg_type,                                 \
                                    kernel_name,                              \
                                    backend,                                  \
                                    context,                                  \
                                    layout,                                   \
                                    registrar_id,                             \
                                    args_def_fn,                              \
                                    meta_kernel_fn,                           \
                                    cpp_dtype,                                \
                                    ...)                                      \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_8(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_10(reg_type,                                \
                                     kernel_name,                             \
                                     backend,                                 \
                                     context,                                 \
                                     layout,                                  \
                                     registrar_id,                            \
                                     args_def_fn,                             \
                                     meta_kernel_fn,                          \
                                     cpp_dtype,                               \
                                     ...)                                     \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_9(reg_type,                             \
                                        kernel_name,                          \
                                        backend,                              \
                                        context,                              \
                                        layout,                               \
                                        PT_ID,                                \
                                        args_def_fn,                          \
                                        meta_kernel_fn,                       \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_11(reg_type,                                \
                                     kernel_name,                             \
                                     backend,                                 \
                                     context,                                 \
                                     layout,                                  \
                                     registrar_id,                            \
                                     args_def_fn,                             \
                                     meta_kernel_fn,                          \
                                     cpp_dtype,                               \
                                     ...)                                     \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_10(reg_type,                            \
                                         kernel_name,                         \
                                         backend,                             \
                                         context,                             \
                                         layout,                              \
                                         PT_ID,                               \
                                         args_def_fn,                         \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_12(reg_type,                                \
                                     kernel_name,                             \
                                     backend,                                 \
                                     context,                                 \
                                     layout,                                  \
                                     registrar_id,                            \
                                     args_def_fn,                             \
                                     meta_kernel_fn,                          \
                                     cpp_dtype,                               \
                                     ...)                                     \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_11(reg_type,                            \
                                         kernel_name,                         \
                                         backend,                             \
                                         context,                             \
                                         layout,                              \
                                         PT_ID,                               \
                                         args_def_fn,                         \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_13(reg_type,                                \
                                     kernel_name,                             \
                                     backend,                                 \
                                     context,                                 \
                                     layout,                                  \
                                     registrar_id,                            \
                                     args_def_fn,                             \
                                     meta_kernel_fn,                          \
                                     cpp_dtype,                               \
                                     ...)                                     \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_12(reg_type,                            \
                                         kernel_name,                         \
                                         backend,                             \
                                         context,                             \
                                         layout,                              \
                                         PT_ID,                               \
                                         args_def_fn,                         \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_14(reg_type,                                \
                                     kernel_name,                             \
                                     backend,                                 \
                                     context,                                 \
                                     layout,                                  \
                                     registrar_id,                            \
                                     args_def_fn,                             \
                                     meta_kernel_fn,                          \
                                     cpp_dtype,                               \
                                     ...)                                     \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_13(reg_type,                            \
                                         kernel_name,                         \
                                         backend,                             \
                                         context,                             \
                                         layout,                              \
                                         PT_ID,                               \
                                         args_def_fn,                         \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_15(reg_type,                                \
                                     kernel_name,                             \
                                     backend,                                 \
                                     context,                                 \
                                     layout,                                  \
                                     registrar_id,                            \
                                     args_def_fn,                             \
                                     meta_kernel_fn,                          \
                                     cpp_dtype,                               \
                                     ...)                                     \
  static const ::phi::KernelRegistrar PT_CONCATENATE(                         \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                               \
      #kernel_name,                                                           \
      #backend,                                                               \
      DATALAYOUT(layout),                                                     \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(),           \
      ::phi::KernelArgsParseFunctor<decltype(                                 \
          &meta_kernel_fn<cpp_dtype, context>)>::Parse,                       \
      args_def_fn,                                                            \
      PT_KERNEL(meta_kernel_fn<cpp_dtype, context>),                          \
      PT_VARIADIC_KERNEL(meta_kernel_fn<cpp_dtype, context>));                \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_14(reg_type,                            \
                                         kernel_name,                         \
                                         backend,                             \
                                         context,                             \
                                         layout,                              \
                                         PT_ID,                               \
                                         args_def_fn,                         \
                                         meta_kernel_fn,                      \
                                         __VA_ARGS__))
/** PD_REGISTER_GENERAL_KERNEL
 *
 * Basic Kernel register marco, used to register a instantiated kernel function
 * with one template argument.
 */

#define PD_REGISTER_GENERAL_KERNEL(                 \
    kernel_name, backend, layout, kernel_fn, dtype) \
  _PD_REGISTER_GENERAL_KERNEL(                      \
      ::phi::RegType::INNER, kernel_name, backend, layout, kernel_fn, dtype)

#define _PD_REGISTER_GENERAL_KERNEL(                                         \
    reg_type, kernel_name, backend, layout, kernel_fn, dtype)                \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      PD_REGISTER_no_t_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_REGISTER_NO_TEMPLATE_KERNEL must be called in global namespace."); \
  __PD_REGISTER_GENERAL_KERNEL(                                              \
      reg_type, kernel_name, backend, layout, kernel_fn, dtype)

#ifndef _WIN32
#define __PD_REGISTER_GENERAL_KERNEL(                                       \
    reg_type, kernel_name, backend, layout, kernel_fn, dtype)               \
  template decltype(kernel_fn) kernel_fn;                                   \
  static void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
  static const ::phi::KernelRegistrar                                       \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout(                 \
          reg_type,                                                         \
          #kernel_name,                                                     \
          #backend,                                                         \
          DATALAYOUT(layout),                                               \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,       \
          &__PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,    \
          PT_KERNEL(kernel_fn),                                             \
          PT_VARIADIC_KERNEL(kernel_fn));                                   \
  int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() {         \
    return 0;                                                               \
  }                                                                         \
  void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
#else
#define __PD_REGISTER_GENERAL_KERNEL(                                       \
    reg_type, kernel_name, backend, layout, kernel_fn, dtype)               \
  static void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
  static const ::phi::KernelRegistrar                                       \
      __reg_pt_kernel_##kernel_name##_##backend##_##layout(                 \
          reg_type,                                                         \
          #kernel_name,                                                     \
          #backend,                                                         \
          DATALAYOUT(layout),                                               \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,       \
          &__PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,    \
          PT_KERNEL(kernel_fn),                                             \
          PT_VARIADIC_KERNEL(kernel_fn));                                   \
  int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() {         \
    return 0;                                                               \
  }                                                                         \
  void __PT_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
#endif

/** PD_DECLARE_KERNEL
 *
 * Used to export the symbols of the file where the kernel is located,
 * to avoid being removed by linker
 */
#define PD_DECLARE_KERNEL(kernel_name, backend, layout)                   \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      PD_DECLARE_tp_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_DECLARE_KERNEL must be called in global namespace.");           \
  extern int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout(); \
  UNUSED static int                                                       \
      __declare_kernel_symbol_for_##kernel_name##_##backend##_##layout =  \
          TouchKernelSymbolFor_##kernel_name##_##backend##_##layout()

/** PD_REGISTER_BUILTIN_KERNEL
 *
 * Used to register kernels for built-in backends.
 * Support CPU GPU XPU.
 */
#define PD_REGISTER_BUILTIN_KERNEL(                    \
    kernel_name, backend, layout, meta_kernel_fn, ...) \
  _PD_REGISTER_KERNEL(::phi::RegType::OUTER,           \
                      kernel_name,                     \
                      backend,                         \
                      ::phi::backend##Context,         \
                      layout,                          \
                      meta_kernel_fn,                  \
                      __VA_ARGS__)

/** PD_REGISTER_PLUGIN_KERNEL
 *
 * Used to register kernels for plug-in backends.
 * Support user-defined backend such as 'Ascend910'.
 */
#define PD_REGISTER_PLUGIN_KERNEL(                     \
    kernel_name, backend, layout, meta_kernel_fn, ...) \
  _PD_REGISTER_KERNEL(::phi::RegType::OUTER,           \
                      kernel_name,                     \
                      backend,                         \
                      ::phi::CustomContext,            \
                      layout,                          \
                      meta_kernel_fn,                  \
                      __VA_ARGS__)

}  // namespace phi
