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

#include <typeindex>
#include <typeinfo>

#include "paddle/phi/core/custom_kernel.h"
#include "paddle/phi/core/kernel_utils.h"

namespace phi {

#define BACKEND(arg__) phi::Backend::arg__
#define DATA_LAYOUT(arg__) phi::DataLayout::arg__
#define DATATYPE(arg__) phi::DataType::arg__

template <typename Func>
struct KernelArgsParseFunctor;

void SetKernelArgsDef(const std::vector<std::type_index>& args_type,
                      const KernelKey& default_key,
                      KernelArgsDef* args_def);

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

    auto args_type = ParseArgType(Indices{});
    SetKernelArgsDef(args_type, default_key, args_def);
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
      // NOTE(zhoushunjie): Only the strings kernels can support pstring dtype
      constexpr char strings_kernels_prefix[] = "strings_";
      if (dtype == static_cast<size_t>(DataType::PSTRING) &&
          strncmp(kernel_name_cstr,
                  strings_kernels_prefix,
                  strlen(strings_kernels_prefix))) {
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
    if (kernel.GetKernelRegisteredType() == KernelRegisteredType::FUNCTION) {
      args_parse_fn(kernel_key, kernel.mutable_args_def());
    }
    args_def_fn(kernel_key, &kernel);
    if (reg_type == RegType::INNER) {
      KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
    } else {
      CustomKernelMap::Instance().RegisterCustomKernel(
          kernel_name, kernel_key, kernel);
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
#define PD_NARGS(...) _PD_NARGS((__VA_ARGS__, _PD_RESQ_N()))
#define _PD_NARGS(...) _PD_ARG_N(__VA_ARGS__)
#define _PD_ARG_N_EXPAND(                                                     \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) \
  N
#define _PD_ARG_N(args) _PD_ARG_N_EXPAND args
#define _PD_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

// The macro for passing KernelArgsParseFunctor's function
#define ARG_PARSE_FUNCTOR(meta_kernel_fn, cpp_dtype, context) \
  ::phi::KernelArgsParseFunctor<                              \
      decltype(&meta_kernel_fn<cpp_dtype, context>)>::Parse

// The macro for instantiating function kernel
#define FUNCTION_KERNEL_INSTANTIATION(meta_kernel_fn, cpp_dtype, context) \
  template decltype(meta_kernel_fn<cpp_dtype, context>)                   \
      meta_kernel_fn<cpp_dtype, context>;

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
                      FUNCTION_KERNEL_INSTANTIATION,                          \
                      ARG_PARSE_FUNCTOR,                                      \
                      PHI_KERNEL,                                             \
                      PHI_VARIADIC_KERNEL,                                    \
                      __VA_ARGS__)

#define _PD_REGISTER_KERNEL(reg_type,                                      \
                            kernel_name,                                   \
                            backend,                                       \
                            context,                                       \
                            layout,                                        \
                            meta_kernel_fn,                                \
                            kernel_instantiation_macro,                    \
                            arg_parse_functor_macro,                       \
                            kernel_unfold_macro,                           \
                            variadic_kernel_unfold_marco,                  \
                            ...)                                           \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                       \
      PD_REGISTER_tp_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_REGISTER_KERNEL must be called in global namespace.");           \
  PD_EXPAND(_PD_REGISTER_2TA_KERNEL(reg_type,                              \
                                    kernel_name,                           \
                                    backend,                               \
                                    context,                               \
                                    layout,                                \
                                    meta_kernel_fn,                        \
                                    kernel_instantiation_macro,            \
                                    arg_parse_functor_macro,               \
                                    kernel_unfold_macro,                   \
                                    variadic_kernel_unfold_marco,          \
                                    __VA_ARGS__))

#ifndef _WIN32
#define _PD_REGISTER_2TA_KERNEL(reg_type,                                   \
                                kernel_name,                                \
                                backend,                                    \
                                context,                                    \
                                layout,                                     \
                                meta_kernel_fn,                             \
                                kernel_instantiation_macro,                 \
                                arg_parse_functor_macro,                    \
                                kernel_unfold_macro,                        \
                                variadic_kernel_unfold_marco,               \
                                ...)                                        \
  PD_KERNEL_INSTANTIATION(meta_kernel_fn,                                   \
                          backend,                                          \
                          context,                                          \
                          kernel_instantiation_macro,                       \
                          __VA_ARGS__);                                     \
  static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
  PD_KERNEL_REGISTRAR_INIT(                                                 \
      reg_type,                                                             \
      kernel_name,                                                          \
      backend,                                                              \
      context,                                                              \
      layout,                                                               \
      &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,        \
      meta_kernel_fn,                                                       \
      arg_parse_functor_macro,                                              \
      kernel_unfold_macro,                                                  \
      variadic_kernel_unfold_marco,                                         \
      __VA_ARGS__);                                                         \
  void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
      const ::phi::KernelKey& kernel_key UNUSED, ::phi::Kernel* kernel UNUSED)
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
#define _PD_REGISTER_2TA_KERNEL(reg_type,                                   \
                                kernel_name,                                \
                                backend,                                    \
                                context,                                    \
                                layout,                                     \
                                meta_kernel_fn,                             \
                                kernel_instantiation_macro,                 \
                                arg_parse_functor_macro,                    \
                                kernel_unfold_macro,                        \
                                variadic_kernel_unfold_marco,               \
                                ...)                                        \
  static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
  PD_EXPAND(PD_KERNEL_REGISTRAR_INIT(                                       \
      reg_type,                                                             \
      kernel_name,                                                          \
      backend,                                                              \
      context,                                                              \
      layout,                                                               \
      &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,        \
      meta_kernel_fn,                                                       \
      arg_parse_functor_macro,                                              \
      kernel_unfold_macro,                                                  \
      variadic_kernel_unfold_marco,                                         \
      __VA_ARGS__));                                                        \
  void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
#endif

#define PD_KERNEL_INSTANTIATION(                                       \
    meta_kernel_fn, backend, context, kernel_instantiation_macro, ...) \
  _PD_KERNEL_INSTANTIATION(PD_NARGS(__VA_ARGS__),                      \
                           meta_kernel_fn,                             \
                           backend,                                    \
                           context,                                    \
                           kernel_instantiation_macro,                 \
                           __VA_ARGS__)

#define _PD_KERNEL_INSTANTIATION(                                         \
    N, meta_kernel_fn, backend, context, kernel_instantiation_macro, ...) \
  PD_CONCATENATE(_PD_KERNEL_INSTANTIATION_, N)                            \
  (meta_kernel_fn, backend, context, kernel_instantiation_macro, __VA_ARGS__)

#define _PD_KERNEL_INSTANTIATION_1(                                          \
    meta_kernel_fn, backend, context, kernel_instantiation_macro, cpp_dtype) \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)
#define _PD_KERNEL_INSTANTIATION_2(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_1(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_3(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_2(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_4(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_3(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_5(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_4(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_6(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_5(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_7(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_6(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_8(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_7(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_9(meta_kernel_fn,                     \
                                   backend,                            \
                                   context,                            \
                                   kernel_instantiation_macro,         \
                                   cpp_dtype,                          \
                                   ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_8(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_10(meta_kernel_fn,                    \
                                    backend,                           \
                                    context,                           \
                                    kernel_instantiation_macro,        \
                                    cpp_dtype,                         \
                                    ...)                               \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)       \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_9(meta_kernel_fn,             \
                                           backend,                    \
                                           context,                    \
                                           kernel_instantiation_macro, \
                                           __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_11(meta_kernel_fn,                     \
                                    backend,                            \
                                    context,                            \
                                    kernel_instantiation_macro,         \
                                    cpp_dtype,                          \
                                    ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)        \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_10(meta_kernel_fn,             \
                                            backend,                    \
                                            context,                    \
                                            kernel_instantiation_macro, \
                                            __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_12(meta_kernel_fn,                     \
                                    backend,                            \
                                    context,                            \
                                    kernel_instantiation_macro,         \
                                    cpp_dtype,                          \
                                    ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)        \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_11(meta_kernel_fn,             \
                                            backend,                    \
                                            context,                    \
                                            kernel_instantiation_macro, \
                                            __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_13(meta_kernel_fn,                     \
                                    backend,                            \
                                    context,                            \
                                    kernel_instantiation_macro,         \
                                    cpp_dtype,                          \
                                    ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)        \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_12(meta_kernel_fn,             \
                                            backend,                    \
                                            context,                    \
                                            kernel_instantiation_macro, \
                                            __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_14(meta_kernel_fn,                     \
                                    backend,                            \
                                    context,                            \
                                    kernel_instantiation_macro,         \
                                    cpp_dtype,                          \
                                    ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)        \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_13(meta_kernel_fn,             \
                                            backend,                    \
                                            context,                    \
                                            kernel_instantiation_macro, \
                                            __VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_15(meta_kernel_fn,                     \
                                    backend,                            \
                                    context,                            \
                                    kernel_instantiation_macro,         \
                                    cpp_dtype,                          \
                                    ...)                                \
  kernel_instantiation_macro(meta_kernel_fn, cpp_dtype, context)        \
      PD_EXPAND(_PD_KERNEL_INSTANTIATION_14(meta_kernel_fn,             \
                                            backend,                    \
                                            context,                    \
                                            kernel_instantiation_macro, \
                                            __VA_ARGS__))

#define PD_KERNEL_REGISTRAR_INIT(reg_type,                          \
                                 kernel_name,                       \
                                 backend,                           \
                                 context,                           \
                                 layout,                            \
                                 args_def_fn,                       \
                                 meta_kernel_fn,                    \
                                 arg_parse_functor_macro,           \
                                 kernel_unfold_macro,               \
                                 variadic_kernel_unfold_marco,      \
                                 ...)                               \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT(PD_NARGS(__VA_ARGS__),        \
                                      reg_type,                     \
                                      kernel_name,                  \
                                      backend,                      \
                                      context,                      \
                                      layout,                       \
                                      args_def_fn,                  \
                                      meta_kernel_fn,               \
                                      arg_parse_functor_macro,      \
                                      kernel_unfold_macro,          \
                                      variadic_kernel_unfold_marco, \
                                      __VA_ARGS__))

// clang-format off

/* The =pre-commit always treats this macro into the wrong format,
  and multi-line macros cannot be skipped with NOLINT.*/
#define _PD_KERNEL_REGISTRAR_INIT(N,                       \
                                  reg_type,                \
                                  kernel_name,             \
                                  backend,                 \
                                  context,                 \
                                  layout,                  \
                                  args_def_fn,             \
                                  meta_kernel_fn,          \
                                  arg_parse_functor_macro,       \
                                  kernel_unfold_macro,               \
                                  variadic_kernel_unfold_marco,      \
                                  ...)                     \
  PD_EXPAND(PD_CONCATENATE(_PD_KERNEL_REGISTRAR_INIT_, N) ( \
    reg_type,                                              \
    kernel_name,                                           \
    backend,                                               \
    context,                                               \
    layout,                                                \
    PD_ID,                                                 \
    args_def_fn,                                           \
    meta_kernel_fn,                                        \
    arg_parse_functor_macro,                                     \
    kernel_unfold_macro,                                             \
    variadic_kernel_unfold_marco,                                    \
    __VA_ARGS__))

// clang-format on

#define _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                  \
                                    kernel_name,                               \
                                    backend,                                   \
                                    context,                                   \
                                    layout,                                    \
                                    registrar_id,                              \
                                    args_def_fn,                               \
                                    meta_kernel_fn,                            \
                                    arg_parse_functor_macro,                   \
                                    kernel_unfold_macro,                       \
                                    variadic_kernel_unfold_marco,              \
                                    cpp_dtype)                                 \
  static const ::phi::KernelRegistrar PD_CONCATENATE(                          \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
      reg_type,                                                                \
      #kernel_name,                                                            \
      #backend,                                                                \
      DATA_LAYOUT(layout),                                                     \
      ::phi::CppTypeToDataType<cpp_dtype>::Type(),                             \
      arg_parse_functor_macro(meta_kernel_fn, cpp_dtype, context),             \
      args_def_fn,                                                             \
      kernel_unfold_macro(meta_kernel_fn<cpp_dtype, context>),                 \
      variadic_kernel_unfold_marco(meta_kernel_fn<cpp_dtype, context>));

#define _PD_KERNEL_REGISTRAR_INIT_1(reg_type,                                \
                                    kernel_name,                             \
                                    backend,                                 \
                                    context,                                 \
                                    layout,                                  \
                                    registrar_id,                            \
                                    args_def_fn,                             \
                                    meta_kernel_fn,                          \
                                    arg_parse_functor_macro,                 \
                                    kernel_unfold_macro,                     \
                                    variadic_kernel_unfold_marco,            \
                                    cpp_dtype)                               \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                      \
                              kernel_name,                                   \
                              backend,                                       \
                              context,                                       \
                              layout,                                        \
                              registrar_id,                                  \
                              args_def_fn,                                   \
                              meta_kernel_fn,                                \
                              arg_parse_functor_macro,                       \
                              kernel_unfold_macro,                           \
                              variadic_kernel_unfold_marco,                  \
                              cpp_dtype)                                     \
  TEST_API int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
    return 0;                                                                \
  }
#define _PD_KERNEL_REGISTRAR_INIT_2(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_1(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_3(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_2(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_4(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_3(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_5(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_4(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_6(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_5(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_7(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_6(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_8(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_7(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_9(reg_type,                         \
                                    kernel_name,                      \
                                    backend,                          \
                                    context,                          \
                                    layout,                           \
                                    registrar_id,                     \
                                    args_def_fn,                      \
                                    meta_kernel_fn,                   \
                                    arg_parse_functor_macro,          \
                                    kernel_unfold_macro,              \
                                    variadic_kernel_unfold_marco,     \
                                    cpp_dtype,                        \
                                    ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_8(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_10(reg_type,                        \
                                     kernel_name,                     \
                                     backend,                         \
                                     context,                         \
                                     layout,                          \
                                     registrar_id,                    \
                                     args_def_fn,                     \
                                     meta_kernel_fn,                  \
                                     arg_parse_functor_macro,         \
                                     kernel_unfold_macro,             \
                                     variadic_kernel_unfold_marco,    \
                                     cpp_dtype,                       \
                                     ...)                             \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                              kernel_name,                            \
                              backend,                                \
                              context,                                \
                              layout,                                 \
                              registrar_id,                           \
                              args_def_fn,                            \
                              meta_kernel_fn,                         \
                              arg_parse_functor_macro,                \
                              kernel_unfold_macro,                    \
                              variadic_kernel_unfold_marco,           \
                              cpp_dtype)                              \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_9(reg_type,                     \
                                        kernel_name,                  \
                                        backend,                      \
                                        context,                      \
                                        layout,                       \
                                        PD_ID,                        \
                                        args_def_fn,                  \
                                        meta_kernel_fn,               \
                                        arg_parse_functor_macro,      \
                                        kernel_unfold_macro,          \
                                        variadic_kernel_unfold_marco, \
                                        __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_11(reg_type,                         \
                                     kernel_name,                      \
                                     backend,                          \
                                     context,                          \
                                     layout,                           \
                                     registrar_id,                     \
                                     args_def_fn,                      \
                                     meta_kernel_fn,                   \
                                     arg_parse_functor_macro,          \
                                     kernel_unfold_macro,              \
                                     variadic_kernel_unfold_marco,     \
                                     cpp_dtype,                        \
                                     ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                \
                              kernel_name,                             \
                              backend,                                 \
                              context,                                 \
                              layout,                                  \
                              registrar_id,                            \
                              args_def_fn,                             \
                              meta_kernel_fn,                          \
                              arg_parse_functor_macro,                 \
                              kernel_unfold_macro,                     \
                              variadic_kernel_unfold_marco,            \
                              cpp_dtype)                               \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_10(reg_type,                     \
                                         kernel_name,                  \
                                         backend,                      \
                                         context,                      \
                                         layout,                       \
                                         PD_ID,                        \
                                         args_def_fn,                  \
                                         meta_kernel_fn,               \
                                         arg_parse_functor_macro,      \
                                         kernel_unfold_macro,          \
                                         variadic_kernel_unfold_marco, \
                                         __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_12(reg_type,                         \
                                     kernel_name,                      \
                                     backend,                          \
                                     context,                          \
                                     layout,                           \
                                     registrar_id,                     \
                                     args_def_fn,                      \
                                     meta_kernel_fn,                   \
                                     arg_parse_functor_macro,          \
                                     kernel_unfold_macro,              \
                                     variadic_kernel_unfold_marco,     \
                                     cpp_dtype,                        \
                                     ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                \
                              kernel_name,                             \
                              backend,                                 \
                              context,                                 \
                              layout,                                  \
                              registrar_id,                            \
                              args_def_fn,                             \
                              meta_kernel_fn,                          \
                              arg_parse_functor_macro,                 \
                              kernel_unfold_macro,                     \
                              variadic_kernel_unfold_marco,            \
                              cpp_dtype)                               \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_11(reg_type,                     \
                                         kernel_name,                  \
                                         backend,                      \
                                         context,                      \
                                         layout,                       \
                                         PD_ID,                        \
                                         args_def_fn,                  \
                                         meta_kernel_fn,               \
                                         arg_parse_functor_macro,      \
                                         kernel_unfold_macro,          \
                                         variadic_kernel_unfold_marco, \
                                         __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_13(reg_type,                         \
                                     kernel_name,                      \
                                     backend,                          \
                                     context,                          \
                                     layout,                           \
                                     registrar_id,                     \
                                     args_def_fn,                      \
                                     meta_kernel_fn,                   \
                                     arg_parse_functor_macro,          \
                                     kernel_unfold_macro,              \
                                     variadic_kernel_unfold_marco,     \
                                     cpp_dtype,                        \
                                     ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                \
                              kernel_name,                             \
                              backend,                                 \
                              context,                                 \
                              layout,                                  \
                              registrar_id,                            \
                              args_def_fn,                             \
                              meta_kernel_fn,                          \
                              arg_parse_functor_macro,                 \
                              kernel_unfold_macro,                     \
                              variadic_kernel_unfold_marco,            \
                              cpp_dtype)                               \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_12(reg_type,                     \
                                         kernel_name,                  \
                                         backend,                      \
                                         context,                      \
                                         layout,                       \
                                         PD_ID,                        \
                                         args_def_fn,                  \
                                         meta_kernel_fn,               \
                                         arg_parse_functor_macro,      \
                                         kernel_unfold_macro,          \
                                         variadic_kernel_unfold_marco, \
                                         __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_14(reg_type,                         \
                                     kernel_name,                      \
                                     backend,                          \
                                     context,                          \
                                     layout,                           \
                                     registrar_id,                     \
                                     args_def_fn,                      \
                                     meta_kernel_fn,                   \
                                     arg_parse_functor_macro,          \
                                     kernel_unfold_macro,              \
                                     variadic_kernel_unfold_marco,     \
                                     cpp_dtype,                        \
                                     ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                \
                              kernel_name,                             \
                              backend,                                 \
                              context,                                 \
                              layout,                                  \
                              registrar_id,                            \
                              args_def_fn,                             \
                              meta_kernel_fn,                          \
                              arg_parse_functor_macro,                 \
                              kernel_unfold_macro,                     \
                              variadic_kernel_unfold_marco,            \
                              cpp_dtype)                               \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_13(reg_type,                     \
                                         kernel_name,                  \
                                         backend,                      \
                                         context,                      \
                                         layout,                       \
                                         PD_ID,                        \
                                         args_def_fn,                  \
                                         meta_kernel_fn,               \
                                         arg_parse_functor_macro,      \
                                         kernel_unfold_macro,          \
                                         variadic_kernel_unfold_marco, \
                                         __VA_ARGS__))
#define _PD_KERNEL_REGISTRAR_INIT_15(reg_type,                         \
                                     kernel_name,                      \
                                     backend,                          \
                                     context,                          \
                                     layout,                           \
                                     registrar_id,                     \
                                     args_def_fn,                      \
                                     meta_kernel_fn,                   \
                                     arg_parse_functor_macro,          \
                                     kernel_unfold_macro,              \
                                     variadic_kernel_unfold_marco,     \
                                     cpp_dtype,                        \
                                     ...)                              \
  _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                \
                              kernel_name,                             \
                              backend,                                 \
                              context,                                 \
                              layout,                                  \
                              registrar_id,                            \
                              args_def_fn,                             \
                              meta_kernel_fn,                          \
                              arg_parse_functor_macro,                 \
                              kernel_unfold_macro,                     \
                              variadic_kernel_unfold_marco,            \
                              cpp_dtype)                               \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_14(reg_type,                     \
                                         kernel_name,                  \
                                         backend,                      \
                                         context,                      \
                                         layout,                       \
                                         PD_ID,                        \
                                         args_def_fn,                  \
                                         meta_kernel_fn,               \
                                         arg_parse_functor_macro,      \
                                         kernel_unfold_macro,          \
                                         variadic_kernel_unfold_marco, \
                                         __VA_ARGS__))
/** PD_REGISTER_KERNEL_FOR_ALL_DTYPE
 *
 * Basic Kernel register marco, used to register a instantiated kernel function
 * with one template argument.
 */

#define PD_REGISTER_KERNEL_FOR_ALL_DTYPE(    \
    kernel_name, backend, layout, kernel_fn) \
  _PD_REGISTER_KERNEL_FOR_ALL_DTYPE(         \
      ::phi::RegType::INNER, kernel_name, backend, layout, kernel_fn)

#define _PD_REGISTER_KERNEL_FOR_ALL_DTYPE(                                   \
    reg_type, kernel_name, backend, layout, kernel_fn)                       \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      PD_REGISTER_no_t_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_REGISTER_NO_TEMPLATE_KERNEL must be called in global namespace."); \
  __PD_REGISTER_KERNEL_FOR_ALL_DTYPE(                                        \
      reg_type, kernel_name, backend, layout, kernel_fn)

#ifndef _WIN32
#define __PD_REGISTER_KERNEL_FOR_ALL_DTYPE(                                  \
    reg_type, kernel_name, backend, layout, kernel_fn)                       \
  template decltype(kernel_fn) kernel_fn;                                    \
  static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(  \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);            \
  static const ::phi::KernelRegistrar                                        \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout(                 \
          reg_type,                                                          \
          #kernel_name,                                                      \
          #backend,                                                          \
          DATA_LAYOUT(layout),                                               \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,        \
          &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,     \
          PHI_KERNEL(kernel_fn),                                             \
          PHI_VARIADIC_KERNEL(kernel_fn));                                   \
  TEST_API int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
    return 0;                                                                \
  }                                                                          \
  void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(         \
      const ::phi::KernelKey& kernel_key UNUSED, ::phi::Kernel* kernel UNUSED)
#else
#define __PD_REGISTER_KERNEL_FOR_ALL_DTYPE(                                  \
    reg_type, kernel_name, backend, layout, kernel_fn)                       \
  static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(  \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);            \
  static const ::phi::KernelRegistrar                                        \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout(                 \
          reg_type,                                                          \
          #kernel_name,                                                      \
          #backend,                                                          \
          DATA_LAYOUT(layout),                                               \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,        \
          &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,     \
          PHI_KERNEL(kernel_fn),                                             \
          PHI_VARIADIC_KERNEL(kernel_fn));                                   \
  TEST_API int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
    return 0;                                                                \
  }                                                                          \
  void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(         \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
#endif

/** PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE
 *
 * Used to register a instantiated kernel function
 * for all backend with one template argument.
 */
#define PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(                  \
    kernel_name, layout, meta_kernel_fn)                           \
  _PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(::phi::RegType::INNER, \
                                            kernel_name,           \
                                            layout,                \
                                            meta_kernel_fn,        \
                                            BACKEND_LIST)

#define PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(    \
    kernel_name, layout, meta_kernel_fn)                           \
  _PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(::phi::RegType::INNER, \
                                            kernel_name,           \
                                            layout,                \
                                            meta_kernel_fn,        \
                                            BACKEND_LIST_EXCEPT_CUSTOM)

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#define _DEVICE GPU,
#elif defined(PADDLE_WITH_XPU)
#define _DEVICE XPU,
#else
#define _DEVICE
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#define _CUSTOM Custom,
#else
#define _CUSTOM
#endif

#define BACKEND_LIST _DEVICE _CUSTOM CPU
#define BACKEND_LIST_EXCEPT_CUSTOM _DEVICE CPU

#define _PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(                            \
    reg_type, kernel_name, layout, meta_kernel_fn, ...)                       \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      PD_REGISTER_nt_kernel_ns_check_##kernel_name##_##layout,                \
      "PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE must be called in global "    \
      "namespace.");                                                          \
  PD_EXPAND(__PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reg_type,              \
                                                       kernel_name,           \
                                                       layout,                \
                                                       meta_kernel_fn,        \
                                                       PD_NARGS(__VA_ARGS__), \
                                                       __VA_ARGS__))

#define __PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(               \
    reg_type, kernel_name, layout, meta_kernel_fn, N, ...)        \
  static void __PD_KERNEL_args_def_FN_##kernel_name##_##layout(   \
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel); \
  PD_EXPAND(PD_CONCATENATE(_PD_FOR_ALL_BACKEND_DTYPE_, N)(        \
      reg_type,                                                   \
      kernel_name,                                                \
      layout,                                                     \
      meta_kernel_fn,                                             \
      __PD_KERNEL_args_def_FN_##kernel_name##_##layout,           \
      __VA_ARGS__) void                                           \
                __PD_KERNEL_args_def_FN_##kernel_name##_##layout( \
                    const ::phi::KernelKey& kernel_key UNUSED,    \
                    ::phi::Kernel* kernel UNUSED))
#ifndef _WIN32
#define ___PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(                         \
    reg_type, kernel_name, backend, layout, kernel_fn, args_def_fn)          \
  template decltype(kernel_fn) kernel_fn;                                    \
  static const ::phi::KernelRegistrar                                        \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout(                 \
          reg_type,                                                          \
          #kernel_name,                                                      \
          #backend,                                                          \
          DATA_LAYOUT(layout),                                               \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,        \
          &args_def_fn,                                                      \
          PHI_KERNEL(kernel_fn),                                             \
          PHI_VARIADIC_KERNEL(kernel_fn));                                   \
  TEST_API int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
    return 0;                                                                \
  }
#else
#define ___PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(                         \
    reg_type, kernel_name, backend, layout, kernel_fn, args_def_fn)          \
  static const ::phi::KernelRegistrar                                        \
      __reg_phi_kernel_##kernel_name##_##backend##_##layout(                 \
          reg_type,                                                          \
          #kernel_name,                                                      \
          #backend,                                                          \
          DATA_LAYOUT(layout),                                               \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,        \
          &args_def_fn,                                                      \
          PHI_KERNEL(kernel_fn),                                             \
          PHI_VARIADIC_KERNEL(kernel_fn));                                   \
  TEST_API int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
    return 0;                                                                \
  }
#endif
#define _PD_FOR_ALL_BACKEND_DTYPE_1(                                     \
    reg_type, kernel_name, layout, meta_kernel_fn, args_def_fn, backend) \
  ___PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(                           \
      reg_type,                                                          \
      kernel_name,                                                       \
      backend,                                                           \
      layout,                                                            \
      meta_kernel_fn<::phi::backend##Context>,                           \
      args_def_fn)

#define _PD_FOR_ALL_BACKEND_DTYPE_2(                                          \
    reg_type, kernel_name, layout, meta_kernel_fn, args_def_fn, backend, ...) \
  ___PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(                                \
      reg_type,                                                               \
      kernel_name,                                                            \
      backend,                                                                \
      layout,                                                                 \
      meta_kernel_fn<::phi::backend##Context>,                                \
      args_def_fn)                                                            \
  PD_EXPAND(_PD_FOR_ALL_BACKEND_DTYPE_1(reg_type,                             \
                                        kernel_name,                          \
                                        layout,                               \
                                        meta_kernel_fn,                       \
                                        args_def_fn,                          \
                                        __VA_ARGS__))
#define _PD_FOR_ALL_BACKEND_DTYPE_3(                                          \
    reg_type, kernel_name, layout, meta_kernel_fn, args_def_fn, backend, ...) \
  ___PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(                                \
      reg_type,                                                               \
      kernel_name,                                                            \
      backend,                                                                \
      layout,                                                                 \
      meta_kernel_fn<::phi::backend##Context>,                                \
      args_def_fn)                                                            \
  PD_EXPAND(_PD_FOR_ALL_BACKEND_DTYPE_2(reg_type,                             \
                                        kernel_name,                          \
                                        layout,                               \
                                        meta_kernel_fn,                       \
                                        args_def_fn,                          \
                                        __VA_ARGS__))

/** PD_DECLARE_KERNEL
 *
 * Used to export the symbols of the file where the kernel is located,
 * to avoid being removed by linker
 */
#define PD_DECLARE_KERNEL(kernel_name, backend, layout)                   \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      PD_DECLARE_tp_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_DECLARE_KERNEL must be called in global namespace.");           \
  TEST_API extern int                                                     \
      TouchKernelSymbolFor_##kernel_name##_##backend##_##layout();        \
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
                      FUNCTION_KERNEL_INSTANTIATION,   \
                      ARG_PARSE_FUNCTOR,               \
                      PHI_KERNEL,                      \
                      PHI_VARIADIC_KERNEL,             \
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
                      FUNCTION_KERNEL_INSTANTIATION,   \
                      ARG_PARSE_FUNCTOR,               \
                      PHI_KERNEL,                      \
                      PHI_VARIADIC_KERNEL,             \
                      __VA_ARGS__)

}  // namespace phi
