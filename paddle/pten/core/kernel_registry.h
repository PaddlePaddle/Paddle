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

#include "paddle/pten/core/kernel_def.h"
#include "paddle/pten/core/kernel_factory.h"
#include "paddle/pten/core/kernel_utils.h"

#include "paddle/fluid/platform/enforce.h"

namespace pten {

#define BACKEND(arg__) pten::Backend::arg__
#define DATALAYOUT(arg__) pten::DataLayout::arg__
#define DATATYPE(arg__) pten::DataType::arg__

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
    auto default_tensor_layout = pten::DataLayout::NCHW;
    if (default_key.layout() != pten::DataLayout::ANY) {
      default_tensor_layout = default_key.layout();
    }
    auto args_type = ParseArgType(Indices{});
    for (auto arg_type : args_type) {
      if (arg_type == std::type_index(typeid(const CPUContext&))
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
          ||
          arg_type == std::type_index(typeid(const CUDAContext&))) {
#else
              ) {
#endif
        // do nothing, skip context arg now
      } else if (arg_type == std::type_index(typeid(const DenseTensor&))) {
        args_def->AppendInput(
            default_key.backend(), default_tensor_layout, default_key.dtype());
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<DenseTensor>&))) {
        args_def->AppendInput(
            default_key.backend(), default_tensor_layout, default_key.dtype());
      } else if (arg_type == std::type_index(typeid(DenseTensor*))) {
        args_def->AppendOutput(
            default_key.backend(), default_tensor_layout, default_key.dtype());
      } else if (arg_type ==
                 std::type_index(typeid(std::vector<DenseTensor*>))) {
        args_def->AppendOutput(
            default_key.backend(), default_tensor_layout, default_key.dtype());
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

struct KernelRegistrar {
 public:
  KernelRegistrar(const char* kernel_name_cstr,
                  Backend backend,
                  DataLayout layout,
                  DataType dtype,
                  KernelArgsParseFn args_parse_fn,
                  KernelArgsDefFn args_def_fn,
                  KernelFn kernel_fn) {
    ConstructKernel(kernel_name_cstr,
                    backend,
                    layout,
                    dtype,
                    args_parse_fn,
                    args_def_fn,
                    kernel_fn);
  }

  KernelRegistrar(const char* kernel_name_cstr,
                  Backend backend,
                  DataLayout layout,
                  KernelArgsParseFn args_parse_fn,
                  KernelArgsDefFn args_def_fn,
                  KernelFn kernel_fn) {
    for (size_t dtype = static_cast<size_t>(DataType::BOOL);
         dtype != static_cast<size_t>(DataType::NUM_DATA_TYPES);
         dtype++) {
      ConstructKernel(kernel_name_cstr,
                      backend,
                      layout,
                      static_cast<DataType>(dtype),
                      args_parse_fn,
                      args_def_fn,
                      kernel_fn);
    }
  }

 private:
  void ConstructKernel(const char* kernel_name_cstr,
                       Backend backend,
                       DataLayout layout,
                       DataType dtype,
                       KernelArgsParseFn args_parse_fn,
                       KernelArgsDefFn args_def_fn,
                       KernelFn kernel_fn) {
    KernelName kernel_name(kernel_name_cstr);
    KernelKey kernel_key(backend, layout, dtype);
    Kernel kernel(kernel_fn);
    args_parse_fn(kernel_key, kernel.mutable_args_def());
    args_def_fn(&kernel);
    KernelFactory::Instance().InsertCompatibleOpType(kernel_name.name());
    KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
  }
};

#define PT_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg) \
  _PT_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)

#define _PT_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                    \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#ifdef __COUNTER__
#define PT_ID __COUNTER__
#else
#define PT_ID __LINE__
#endif

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

#define PT_CONCATENATE(arg1, arg2) PT_CONCATENATE1(arg1, arg2)
#define PT_CONCATENATE1(arg1, arg2) PT_CONCATENATE2(arg1, arg2)
#define PT_CONCATENATE2(arg1, arg2) arg1##arg2
#define PT_EXPAND(x) x

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

/** PT_REGISTER_KERNEL
 *
 * The most frequently used kernel registration macro, used for kernel
 * registration with only data type as template parameter, and the function
 * pointer of the corresponding data type is automatically instantiated
 * during registration.
 */
#define PT_REGISTER_KERNEL(                                       \
    kernel_name, backend, layout, meta_kernel_fn, cpp_dtype, ...) \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                              \
      pt_register_kernel_ns_check_##kernel_name,                  \
      "PT_REGISTER_KERNEL must be called in global namespace.");  \
  _PT_REGISTER_KERNEL(                                            \
      kernel_name, backend, layout, meta_kernel_fn, cpp_dtype, __VA_ARGS__)

#ifndef _WIN32
#define _PT_REGISTER_KERNEL(                                          \
    kernel_name, backend, layout, meta_kernel_fn, cpp_dtype, ...)     \
  PT_KERNEL_INSTANTIATION(meta_kernel_fn, cpp_dtype, __VA_ARGS__);    \
  static void __PT_KERNEL_args_def_FN_##kernel_name(::pten::Kernel*); \
  PT_KERNEL_REGISTRAR_INIT(kernel_name,                               \
                           backend,                                   \
                           layout,                                    \
                           &__PT_KERNEL_args_def_FN_##kernel_name,    \
                           meta_kernel_fn,                            \
                           cpp_dtype,                                 \
                           __VA_ARGS__);                              \
  void __PT_KERNEL_args_def_FN_##kernel_name(::pten::Kernel* kernel)
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
#define _PT_REGISTER_KERNEL(                                          \
    kernel_name, backend, layout, meta_kernel_fn, cpp_dtype, ...)     \
  static void __PT_KERNEL_args_def_FN_##kernel_name(::pten::Kernel*); \
  PT_KERNEL_REGISTRAR_INIT(kernel_name,                               \
                           backend,                                   \
                           layout,                                    \
                           &__PT_KERNEL_args_def_FN_##kernel_name,    \
                           meta_kernel_fn,                            \
                           cpp_dtype,                                 \
                           __VA_ARGS__);                              \
  void __PT_KERNEL_args_def_FN_##kernel_name(::pten::Kernel* kernel)
#endif

#define PT_KERNEL_INSTANTIATION(meta_kernel_fn, cpp_dtype, ...) \
  _PT_KERNEL_INSTANTIATION(PT_NARGS(cpp_dtype, __VA_ARGS__),    \
                           meta_kernel_fn,                      \
                           cpp_dtype,                           \
                           __VA_ARGS__)

#define _PT_KERNEL_INSTANTIATION(N, meta_kernel_fn, cpp_dtype, ...) \
  PT_CONCATENATE(_PT_KERNEL_INSTANTIATION_, N)                      \
  (meta_kernel_fn, cpp_dtype, __VA_ARGS__)

#define _PT_KERNEL_INSTANTIATION_1(meta_kernel_fn, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>
#define _PT_KERNEL_INSTANTIATION_2(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_1(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_3(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_2(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_4(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_3(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_5(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_4(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_6(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_5(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_7(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_6(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_8(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_7(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_9(meta_kernel_fn, cpp_dtype, ...)        \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_8(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_10(meta_kernel_fn, cpp_dtype, ...)       \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_9(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_11(meta_kernel_fn, cpp_dtype, ...)       \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_10(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_12(meta_kernel_fn, cpp_dtype, ...)       \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_11(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_13(meta_kernel_fn, cpp_dtype, ...)       \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_12(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_14(meta_kernel_fn, cpp_dtype, ...)       \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_13(meta_kernel_fn, __VA_ARGS__))
#define _PT_KERNEL_INSTANTIATION_15(meta_kernel_fn, cpp_dtype, ...)       \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  PT_EXPAND(_PT_KERNEL_INSTANTIATION_14(meta_kernel_fn, __VA_ARGS__))

#define PT_KERNEL_REGISTRAR_INIT(                                              \
    kernel_name, backend, layout, args_def_fn, meta_kernel_fn, cpp_dtype, ...) \
  _PT_KERNEL_REGISTRAR_INIT(PT_NARGS(cpp_dtype, __VA_ARGS__),                  \
                            kernel_name,                                       \
                            backend,                                           \
                            layout,                                            \
                            args_def_fn,                                       \
                            meta_kernel_fn,                                    \
                            cpp_dtype,                                         \
                            __VA_ARGS__)

// clang-format off

/* The =pre-commit always treats this macro into the wrong format,
  and multi-line macros cannot be skipped with NOLINT.*/
#define _PT_KERNEL_REGISTRAR_INIT(N,              \
                                  kernel_name,    \
                                  backend,        \
                                  layout,         \
                                  args_def_fn,    \
                                  meta_kernel_fn, \
                                  cpp_dtype,      \
                                  ...)            \
  PT_CONCATENATE(_PT_KERNEL_REGISTRAR_INIT_, N) ( \
    kernel_name,                                  \
    PT_ID,                                        \
    backend,                                      \
    layout,                                       \
    args_def_fn,                                  \
    meta_kernel_fn,                               \
    cpp_dtype,                                    \
    __VA_ARGS__)

// clang-format on

#define _PT_KERNEL_REGISTRAR_INIT_1(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  int TouchKernelSymbolFor_##kernel_name##_##backend() { return 0; }
#define _PT_KERNEL_REGISTRAR_INIT_2(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_1(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_3(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_2(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_4(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_3(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_5(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_4(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_6(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_5(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_7(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_6(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_8(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_7(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_9(kernel_name,                    \
                                    registrar_id,                   \
                                    backend,                        \
                                    layout,                         \
                                    args_def_fn,                    \
                                    meta_kernel_fn,                 \
                                    cpp_dtype,                      \
                                    ...)                            \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_8(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_10(kernel_name,                   \
                                     registrar_id,                  \
                                     backend,                       \
                                     layout,                        \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_9(kernel_name,                \
                                        PT_ID,                      \
                                        backend,                    \
                                        layout,                     \
                                        args_def_fn,                \
                                        meta_kernel_fn,             \
                                        __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_11(kernel_name,                   \
                                     registrar_id,                  \
                                     backend,                       \
                                     layout,                        \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_10(kernel_name,               \
                                         PT_ID,                     \
                                         backend,                   \
                                         layout,                    \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_12(kernel_name,                   \
                                     registrar_id,                  \
                                     backend,                       \
                                     layout,                        \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_11(kernel_name,               \
                                         PT_ID,                     \
                                         backend,                   \
                                         layout,                    \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_13(kernel_name,                   \
                                     registrar_id,                  \
                                     backend,                       \
                                     layout,                        \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_12(kernel_name,               \
                                         PT_ID,                     \
                                         backend,                   \
                                         layout,                    \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_14(kernel_name,                   \
                                     registrar_id,                  \
                                     backend,                       \
                                     layout,                        \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_13(kernel_name,               \
                                         PT_ID,                     \
                                         backend,                   \
                                         layout,                    \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         __VA_ARGS__))
#define _PT_KERNEL_REGISTRAR_INIT_15(kernel_name,                   \
                                     registrar_id,                  \
                                     backend,                       \
                                     layout,                        \
                                     args_def_fn,                   \
                                     meta_kernel_fn,                \
                                     cpp_dtype,                     \
                                     ...)                           \
  static const ::pten::KernelRegistrar PT_CONCATENATE(              \
      __reg_pt_kernel_##kernel_name##_, registrar_id)(              \
      #kernel_name,                                                 \
      BACKEND(backend),                                             \
      DATALAYOUT(layout),                                           \
      ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type(), \
      ::pten::KernelArgsParseFunctor<decltype(                      \
          &meta_kernel_fn<cpp_dtype>)>::Parse,                      \
      args_def_fn,                                                  \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));                        \
  PT_EXPAND(_PT_KERNEL_REGISTRAR_INIT_14(kernel_name,               \
                                         PT_ID,                     \
                                         backend,                   \
                                         layout,                    \
                                         args_def_fn,               \
                                         meta_kernel_fn,            \
                                         __VA_ARGS__))

/** PT_REGISTER_SINGLE_KERNEL
 *
 * Used to register a single kernel, pass in the complete function pointer
 * of the kernel, this registration macro will not do automatic template
 * instantiation.
 */
#define PT_REGISTER_SINGLE_KERNEL(                                           \
    kernel_name, backend, layout, dtype, kernel_fn)                          \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      pt_register_single_kernel_ns_check_##kernel_name,                      \
      "PT_REGISTER_SINGLE_KERNEL must be called in global namespace.");      \
  static void __PT_SINGLE_KERNEL_args_def_FN_##kernel_name(::pten::Kernel*); \
  static const ::pten::KernelRegistrar __reg_pt_single_kernel_##kernel_name( \
      #kernel_name,                                                          \
      BACKEND(backend),                                                      \
      DATALAYOUT(layout),                                                    \
      DATATYPE(dtype),                                                       \
      ::pten::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,           \
      args_def_fn,                                                           \
      PT_KERNEL(kernel_fn));                                                 \
  int TouchKernelSymbolFor_##kernel_name##_##backend() { return 0; }         \
  void __PT_SINGLE_KERNEL_args_def_FN_##kernel_name(::pten::Kernel*)

/** PT_REGISTER_KERNEL_ALL_DTYPE
 *
 * Used to register a kernel that supports all data types, such as copy and
 * reshape that are not sensitive to data types.
 */
#define PT_REGISTER_KERNEL_ALL_DTYPE(kernel_name, backend, layout, kernel_fn) \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      pt_register_kernel_all_dtype_ns_check_##kernel_name,                    \
      "PT_REGISTER_KERNEL_ALL_DTYPE must be called in global namespace.");    \
  static void __PT_KERNEL_ALL_DTYPE_args_def_FN_##kernel_name(                \
      ::pten::Kernel*);                                                       \
  static const ::pten::KernelRegistrar                                        \
      __reg_pt_kernel_all_dtype_##kernel_name(                                \
          #kernel_name,                                                       \
          BACKEND(backend),                                                   \
          DATALAYOUT(layout),                                                 \
          ::pten::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,        \
          &__PT_KERNEL_ALL_DTYPE_args_def_FN_##kernel_name,                   \
          PT_KERNEL(kernel_fn));                                              \
  int TouchKernelSymbolFor_##kernel_name##_##backend() { return 0; }          \
  void __PT_KERNEL_ALL_DTYPE_args_def_FN_##kernel_name(::pten::Kernel* kernel)

/** PT_DECLARE_KERNEL
 *
 * Used to export the symbols of the file where the kernel is located,
 * to avoid being removed by linker
 */
#define PT_DECLARE_KERNEL(kernel_name, backend)                             \
  extern int TouchKernelSymbolFor_##kernel_name##_##backend();              \
  UNUSED static int __declare_kernel_symbol_for_##kernel_name##_##backend = \
      TouchKernelSymbolFor_##kernel_name##_##backend()

}  // namespace pten
