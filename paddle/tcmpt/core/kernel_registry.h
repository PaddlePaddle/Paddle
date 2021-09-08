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

#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "paddle/tcmpt/core/kernel_def.h"
#include "paddle/tcmpt/core/kernel_factory.h"
#include "paddle/tcmpt/core/kernel_utils.h"

namespace pt {

#define BACKEND(arg__) pt::Backend::k##arg__
#define DATALAYOUT(arg__) pt::DataLayout::k##arg__
#define DATATYPE(arg__) pt::DataType::k##arg__

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
    auto args_type = ParseArgType(Indices{});
    for (auto arg_type : args_type) {
      if (arg_type == std::type_index(typeid(const DenseTensor&))) {
        args_def->AppendInput(
            default_key.backend(), default_key.layout(), default_key.dtype());
      } else if (arg_type == std::type_index(typeid(DenseTensor*))) {
        args_def->AppendOutput(
            default_key.backend(), default_key.layout(), default_key.dtype());
      } else {
        // TODO(chenweihang): throw argument error
        VLOG(1) << "invalid arg";
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
    KernelName kernel_name(kernel_name_cstr);
    KernelKey kernel_key(backend, layout, dtype);
    Kernel kernel(kernel_fn);
    args_parse_fn(kernel_key, kernel.mutable_args_def());
    args_def_fn(&kernel);

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

#define PT_CONCATENATE(arg1, arg2) PT_CONCATENATE1(arg1, arg2)
#define PT_CONCATENATE1(arg1, arg2) PT_CONCATENATE2(arg1, arg2)
#define PT_CONCATENATE2(arg1, arg2) arg1##arg2

// reference:
// https://stackoverflow.com/questions/1872220/is-it-possible-to-iterate-over-arguments-in-variadic-macros
#define PT_NARGS(...) _PT_NARGS(__VA_ARGS__, _PT_RESQ_N())
#define _PT_NARGS(...) _PT_ARG_N(__VA_ARGS__)
#define _PT_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define _PT_RESQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define PT_REGISTER_KERNEL(                                       \
    kernel_name, backend, layout, meta_kernel_fn, cpp_dtype, ...) \
  _PT_REGISTER_KERNEL(kernel_name,                                \
                      PT_ID,                                      \
                      backend,                                    \
                      layout,                                     \
                      meta_kernel_fn,                             \
                      cpp_dtype,                                  \
                      __VA_ARGS__)

#define _PT_REGISTER_KERNEL(                                                   \
    kernel_name, func_id, backend, layout, meta_kernel_fn, cpp_dtype, ...)     \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      PT_CONCATENATE(pt_op_kernel_ns_check_, func_id),                         \
      "PT_REGISTER_KERNEL must be called in global namespace.");               \
  PT_KERNEL_SPECIALIZE(meta_kernel_fn, cpp_dtype, __VA_ARGS__);                \
  static void PT_CONCATENATE(__PT_KERNEL_args_def_FN_,                         \
                             func_id)(::pt::Kernel*);                          \
  PT_KERNEL_REGISTRAR_INIT(kernel_name,                                        \
                           func_id,                                            \
                           backend,                                            \
                           layout,                                             \
                           &PT_CONCATENATE(__PT_KERNEL_args_def_FN_, func_id), \
                           meta_kernel_fn,                                     \
                           cpp_dtype,                                          \
                           __VA_ARGS__);                                       \
  void PT_CONCATENATE(__PT_KERNEL_args_def_FN_, func_id)(::pt::Kernel * kernel)

#define PT_KERNEL_SPECIALIZE(meta_kernel_fn, cpp_dtype, ...) \
  _PT_KERNEL_SPECIALIZE(PT_NARGS(cpp_dtype, __VA_ARGS__),    \
                        meta_kernel_fn,                      \
                        cpp_dtype,                           \
                        __VA_ARGS__)

#define _PT_KERNEL_SPECIALIZE(N, meta_kernel_fn, cpp_dtype, ...) \
  PT_CONCATENATE(_PT_KERNEL_SPECIALIZE_, N)                      \
  (meta_kernel_fn, cpp_dtype, __VA_ARGS__)

#define _PT_KERNEL_SPECIALIZE_1(meta_kernel_fn, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>
#define _PT_KERNEL_SPECIALIZE_2(meta_kernel_fn, cpp_dtype, ...)           \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  _PT_KERNEL_SPECIALIZE_1(meta_kernel_fn, __VA_ARGS__)
#define _PT_KERNEL_SPECIALIZE_3(meta_kernel_fn, cpp_dtype, ...)           \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  _PT_KERNEL_SPECIALIZE_2(meta_kernel_fn, __VA_ARGS__)
#define _PT_KERNEL_SPECIALIZE_4(meta_kernel_fn, cpp_dtype, ...)           \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  _PT_KERNEL_SPECIALIZE_3(meta_kernel_fn, __VA_ARGS__)
#define _PT_KERNEL_SPECIALIZE_5(meta_kernel_fn, cpp_dtype, ...)           \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  _PT_KERNEL_SPECIALIZE_4(meta_kernel_fn, __VA_ARGS__)
#define _PT_KERNEL_SPECIALIZE_6(meta_kernel_fn, cpp_dtype, ...)           \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  _PT_KERNEL_SPECIALIZE_5(meta_kernel_fn, __VA_ARGS__)
#define _PT_KERNEL_SPECIALIZE_7(meta_kernel_fn, cpp_dtype, ...)           \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  _PT_KERNEL_SPECIALIZE_6(meta_kernel_fn, __VA_ARGS__)
#define _PT_KERNEL_SPECIALIZE_8(meta_kernel_fn, cpp_dtype, ...)           \
  template decltype(meta_kernel_fn<cpp_dtype>) meta_kernel_fn<cpp_dtype>; \
  _PT_KERNEL_SPECIALIZE_7(meta_kernel_fn, __VA_ARGS__)

#define PT_KERNEL_REGISTRAR_INIT(kernel_name,                 \
                                 func_id,                     \
                                 backend,                     \
                                 layout,                      \
                                 args_def_fn,                 \
                                 meta_kernel_fn,              \
                                 cpp_dtype,                   \
                                 ...)                         \
  _PT_KERNEL_REGISTRAR_INIT(PT_NARGS(cpp_dtype, __VA_ARGS__), \
                            kernel_name,                      \
                            func_id,                          \
                            backend,                          \
                            layout,                           \
                            args_def_fn,                      \
                            meta_kernel_fn,                   \
                            cpp_dtype,                        \
                            __VA_ARGS__)

#define _PT_KERNEL_REGISTRAR_INIT(N,                 \
                                  kernel_name,       \
                                  func_id,           \
                                  backend,           \
                                  layout,            \
                                  args_def_fn,       \
                                  meta_kernel_fn,    \
                                  cpp_dtype,         \
                                  ...)               \
  PT_CONCATENATE(_PT_KERNEL_REGISTRAR_INIT_, N)      \
    (kernel_name,                                    \
      func_id,                                       \
      PT_ID,                                         \
      backend,                                       \
      layout,                                        \
      args_def_fn,                                   \
      meta_kernel_fn,                                \
      cpp_dtype,                                     \
      __VA_ARGS__)

#define _PT_KERNEL_REGISTRAR_INIT_1(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));
#define _PT_KERNEL_REGISTRAR_INIT_2(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));          \
  _PT_KERNEL_REGISTRAR_INIT_1(kernel_name,            \
                              func_id,                \
                              PT_ID,                  \
                              backend,                \
                              layout,                 \
                              args_def_fn,            \
                              meta_kernel_fn,         \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_3(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));          \
  _PT_KERNEL_REGISTRAR_INIT_2(kernel_name,            \
                              func_id,                \
                              PT_ID,                  \
                              backend,                \
                              layout,                 \
                              args_def_fn,            \
                              meta_kernel_fn,         \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_4(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));          \
  _PT_KERNEL_REGISTRAR_INIT_3(kernel_name,            \
                              func_id,                \
                              PT_ID,                  \
                              backend,                \
                              layout,                 \
                              args_def_fn,            \
                              meta_kernel_fn,         \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_5(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));          \
  _PT_KERNEL_REGISTRAR_INIT_4(kernel_name,            \
                              func_id,                \
                              PT_ID,                  \
                              backend,                \
                              layout,                 \
                              args_def_fn,            \
                              meta_kernel_fn,         \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_6(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));          \
  _PT_KERNEL_REGISTRAR_INIT_5(kernel_name,            \
                              func_id,                \
                              PT_ID,                  \
                              backend,                \
                              layout,                 \
                              args_def_fn,            \
                              meta_kernel_fn,         \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_7(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));          \
  _PT_KERNEL_REGISTRAR_INIT_6(kernel_name,            \
                              func_id,                \
                              PT_ID,                  \
                              backend,                \
                              layout,                 \
                              args_def_fn,            \
                              meta_kernel_fn,         \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_8(kernel_name,      \
                                    func_id,          \
                                    registrar_id,     \
                                    backend,          \
                                    layout,           \
                                    args_def_fn,      \
                                    meta_kernel_fn,   \
                                    cpp_dtype,        \
                                    ...)              \
  static const ::pt::KernelRegistrar PT_CONCATENATE(  \
      __reg_pt_op_kernel_##func_id##_, registrar_id)( \
      kernel_name,                                    \
      BACKEND(backend),                               \
      DATALAYOUT(layout),                             \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),     \
      ::pt::KernelArgsParseFunctor<decltype(          \
          &meta_kernel_fn<cpp_dtype>)>::Parse,        \
      args_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));          \
  _PT_KERNEL_REGISTRAR_INIT_7(kernel_name,            \
                              func_id,                \
                              PT_ID,                  \
                              backend,                \
                              layout,                 \
                              args_def_fn,            \
                              meta_kernel_fn,         \
                              __VA_ARGS__)

#define PT_REGISTER_KERNEL_STANDARD(                \
    kernel_name, backend, layout, dtype, kernel_fn) \
  _PT_REGISTER_KERNEL_STANDARD(                     \
      kernel_name, PT_ID, backend, layout, dtype, kernel_fn)

#define _PT_REGISTER_KERNEL_STANDARD(                                      \
    kernel_name, func_id, backend, layout, dtype, kernel_fn)               \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                       \
      PT_CONCATENATE(pt_op_kernel_ns_check_, func_id),                     \
      "_PT_REGISTER_KERNEL_STANDARD must be called in global namespace."); \
  template decltype(kernel_fn) kernel_fn;                                  \
  static void PT_CONCATENATE(__PT_KERNEL_args_def_FN_,                     \
                             func_id)(::pt::Kernel*);                      \
  static const ::pt::KernelRegistrar PT_CONCATENATE(__reg_pt_op_kernel_,   \
                                                    func_id)(              \
      kernel_name,                                                         \
      BACKEND(backend),                                                    \
      DATALAYOUT(layout),                                                  \
      DATATYPE(dtype),                                                     \
      ::pt::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,           \
      args_def_fn,                                                         \
      PT_KERNEL(kernel_fn));                                               \
  void PT_CONCATENATE(__PT_KERNEL_args_def_FN_, func_id)(::pt::Kernel*)

}  // namespace pt
