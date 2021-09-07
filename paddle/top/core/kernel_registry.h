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

#include "paddle/top/core/kernel_def.h"
#include "paddle/top/core/kernel_factory.h"
#include "paddle/top/core/kernel_utils.h"

namespace pt {

#define BACKEND(arg__) pt::Backend::k##arg__
#define DATALAYOUT(arg__) pt::DataLayout::k##arg__
#define DATATYPE(arg__) pt::DataType::k##arg__

class OpKernelRegistrar {
 public:
  OpKernelRegistrar(const char* op_name,
                    Backend backend,
                    DataLayout layout,
                    DataType dtype,
                    OpKernelParamDefFn param_def_fn,
                    OpKernelFn kernel_fn) {
    OperationName final_op_name(op_name);
    OpKernelKey op_kernel_key(backend, layout, dtype);
    OpKernel kernel(kernel_fn);
    param_def_fn(&kernel);

    // TODO(chenweihang): use default input and output for verify
    kernel.mutable_param_def()->AppendInput(backend, layout, dtype);
    kernel.mutable_param_def()->AppendOutput(backend, layout, dtype);

    OpKernelFactory::Instance().kernels()[final_op_name][op_kernel_key] =
        kernel;
  }
};

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

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

#define PT_REGISTER_KERNEL(                                   \
    op_name, backend, layout, meta_kernel_fn, cpp_dtype, ...) \
  _PT_REGISTER_KERNEL(                                        \
      op_name, PT_ID, backend, layout, meta_kernel_fn, cpp_dtype, __VA_ARGS__)

#define _PT_REGISTER_KERNEL(                                           \
    op_name, func_id, backend, layout, meta_kernel_fn, cpp_dtype, ...) \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                   \
      PT_CONCATENATE(pt_op_kernel_ns_check_, func_id),                 \
      "PT_REGISTER_KERNEL must be called in global namespace.");       \
  PT_KERNEL_SPECIALIZE(meta_kernel_fn, cpp_dtype, __VA_ARGS__);        \
  static void PT_CONCATENATE(__PT_KERNEL_PARAM_DEF_FN_,                \
                             func_id)(::pt::OpKernel*);                \
  PT_KERNEL_REGISTRAR_INIT(                                            \
      op_name,                                                         \
      func_id,                                                         \
      backend,                                                         \
      layout,                                                          \
      &PT_CONCATENATE(__PT_KERNEL_PARAM_DEF_FN_, func_id),             \
      meta_kernel_fn,                                                  \
      cpp_dtype,                                                       \
      __VA_ARGS__);                                                    \
  void PT_CONCATENATE(__PT_KERNEL_PARAM_DEF_FN_,                       \
                      func_id)(::pt::OpKernel * kernel)

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

#define PT_KERNEL_REGISTRAR_INIT(op_name,                     \
                                 func_id,                     \
                                 backend,                     \
                                 layout,                      \
                                 param_def_fn,                \
                                 meta_kernel_fn,              \
                                 cpp_dtype,                   \
                                 ...)                         \
  _PT_KERNEL_REGISTRAR_INIT(PT_NARGS(cpp_dtype, __VA_ARGS__), \
                            op_name,                          \
                            func_id,                          \
                            backend,                          \
                            layout,                           \
                            param_def_fn,                     \
                            meta_kernel_fn,                   \
                            cpp_dtype,                        \
                            __VA_ARGS__)

#define _PT_KERNEL_REGISTRAR_INIT(N,              \
                                  op_name,        \
                                  func_id,        \
                                  backend,        \
                                  layout,         \
                                  param_def_fn,   \
                                  meta_kernel_fn, \
                                  cpp_dtype,      \
                                  ...)            \
  PT_CONCATENATE(_PT_KERNEL_REGISTRAR_INIT_, N)   \
  (op_name,                                       \
   func_id,                                       \
   PT_ID,                                         \
   backend,                                       \
   layout,                                        \
   param_def_fn,                                  \
   meta_kernel_fn,                                \
   cpp_dtype,                                     \
   __VA_ARGS__)

#define _PT_KERNEL_REGISTRAR_INIT_1(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));
#define _PT_KERNEL_REGISTRAR_INIT_2(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));           \
  _PT_KERNEL_REGISTRAR_INIT_1(op_name,                 \
                              func_id,                 \
                              PT_ID,                   \
                              backend,                 \
                              layout,                  \
                              param_def_fn,            \
                              meta_kernel_fn,          \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_3(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));           \
  _PT_KERNEL_REGISTRAR_INIT_2(op_name,                 \
                              func_id,                 \
                              PT_ID,                   \
                              backend,                 \
                              layout,                  \
                              param_def_fn,            \
                              meta_kernel_fn,          \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_4(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));           \
  _PT_KERNEL_REGISTRAR_INIT_3(op_name,                 \
                              func_id,                 \
                              PT_ID,                   \
                              backend,                 \
                              layout,                  \
                              param_def_fn,            \
                              meta_kernel_fn,          \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_5(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));           \
  _PT_KERNEL_REGISTRAR_INIT_4(op_name,                 \
                              func_id,                 \
                              PT_ID,                   \
                              backend,                 \
                              layout,                  \
                              param_def_fn,            \
                              meta_kernel_fn,          \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_6(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));           \
  _PT_KERNEL_REGISTRAR_INIT_5(op_name,                 \
                              func_id,                 \
                              PT_ID,                   \
                              backend,                 \
                              layout,                  \
                              param_def_fn,            \
                              meta_kernel_fn,          \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_7(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));           \
  _PT_KERNEL_REGISTRAR_INIT_6(op_name,                 \
                              func_id,                 \
                              PT_ID,                   \
                              backend,                 \
                              layout,                  \
                              param_def_fn,            \
                              meta_kernel_fn,          \
                              __VA_ARGS__)
#define _PT_KERNEL_REGISTRAR_INIT_8(op_name,           \
                                    func_id,           \
                                    registrar_id,      \
                                    backend,           \
                                    layout,            \
                                    param_def_fn,      \
                                    meta_kernel_fn,    \
                                    cpp_dtype,         \
                                    ...)               \
  static const ::pt::OpKernelRegistrar PT_CONCATENATE( \
      __reg_pt_op_kernel_##func_id##_, registrar_id)(  \
      op_name,                                         \
      BACKEND(backend),                                \
      DATALAYOUT(layout),                              \
      ::pt::CppTypeToDataType<cpp_dtype>::Type(),      \
      param_def_fn,                                    \
      PT_KERNEL(meta_kernel_fn<cpp_dtype>));           \
  _PT_KERNEL_REGISTRAR_INIT_7(op_name,                 \
                              func_id,                 \
                              PT_ID,                   \
                              backend,                 \
                              layout,                  \
                              param_def_fn,            \
                              meta_kernel_fn,          \
                              __VA_ARGS__)

#define PT_REGISTER_KERNEL_STANDARD(                                      \
    op_name, backend, layout, dtype, kernel_fn)                           \
  template decltype(kernel_fn) kernel_fn;                                 \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      __reg_pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__,  \
      "PT_REGISTER_KERNEL_STANDARD must be called in global namespace."); \
  static ::pt::OpKernelRegistrar                                          \
      __pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__ =     \
          ::pt::OpKernelRegistrar(#op_name,                               \
                                  BACKEND(backend),                       \
                                  DATALAYOUT(layout),                     \
                                  DATATYPE(dtype),                        \
                                  PT_KERNEL(kernel_fn))

#define PT_REGISTER_KERNEL_AUTO_SPECIALIZE(                               \
    op_name, backend, layout, meta_kernel_fn, dtype)                      \
  template decltype(meta_kernel_fn<dtype>) meta_kernel_fn<dtype>;         \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      __reg_pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__,  \
      "PT_REGISTER_KERNEL_AUTO_SPECIALIZE must be called in global "      \
      "namespace.");                                                      \
  static ::pt::OpKernelRegistrar                                          \
      __pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__ =     \
          ::pt::OpKernelRegistrar(#op_name,                               \
                                  BACKEND(backend),                       \
                                  DATALAYOUT(layout),                     \
                                  ::pt::CppTypeToDataType<dtype>::Type(), \
                                  PT_KERNEL(meta_kernel_fn<dtype>))

#define PT_TOUCH_KERNEL_REGISTRAR(op_name, backend, layout, dtype)          \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                        \
      __touch_pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__,  \
      "PT_TOUCH_KERNEL_REGISTRAR must be called in global namespace.");     \
  int TouchOpKernelRegistrar_##op_name##_##backend##_##dtype##_##layout() { \
    __pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__.Touch();  \
    return 0;                                                               \
  }

}  // namespace pt
