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
                    OpKernelFn fn)
      : op_name_(op_name), op_kernel_key_(backend, layout, dtype) {
    OpKernel kernel(fn);
    OpKernelFactory::Instance().kernels()[op_name_][op_kernel_key_] = kernel;
  }

  OpKernelRegistrar& Input(Backend backend, DataLayout layout, DataType dtype) {
    OpKernelFactory::Instance()
        .kernels()[op_name_][op_kernel_key_]
        .mutable_param_def()
        ->AppendInput(backend, layout, dtype);
    return *this;
  }

  OpKernelRegistrar& Output(Backend backend,
                            DataLayout layout,
                            DataType dtype) {
    OpKernelFactory::Instance()
        .kernels()[op_name_][op_kernel_key_]
        .mutable_param_def()
        ->AppendOutput(backend, layout, dtype);
    return *this;
  }

  void Touch() {}

 private:
  OperationName op_name_;
  OpKernelKey op_kernel_key_;
};

#define PT_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                     \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define PT_REGISTER_STANDARD_KERNEL(                                      \
    op_name, backend, layout, dtype, kernel_fn)                           \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      __reg_pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__,  \
      "PT_REGISTER_STANDARD_KERNEL must be called in global namespace."); \
  static ::pt::OpKernelRegistrar                                          \
      __pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__ =     \
          ::pt::OpKernelRegistrar(#op_name,                               \
                                  BACKEND(backend),                       \
                                  DATALAYOUT(layout),                     \
                                  DATATYPE(dtype),                        \
                                  kernel_fn)

#define PT_REGISTER_KERNEL_AUTO_SPECIALIZE(                               \
    op_name, backend, layout, meta_kernel_fn, dtype)                      \
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

/**
 * In most cases, the backend, dtype and layout of Op's input and output
 * are the same as OpKernel itself. In order to simplify the registration
 * writing, we provide the following simple kernel registration macro.
 * If it is an special case, please use PT_REGISTER_STANDARD_KERNEL
 */
// TODO(chenweihang): only work for single input and output now.
// can we use function traits here to parse the input and output type?
#define PT_REGISTER_KERNEL_1T(op_name, backend, layout, meta_kernel_fn, dtype) \
  PT_REGISTER_KERNEL_AUTO_SPECIALIZE(                                          \
      op_name, backend, layout, meta_kernel_fn, dtype)                         \
      .Input(BACKEND(backend),                                                 \
             DATALAYOUT(layout),                                               \
             ::pt::CppTypeToDataType<dtype>::Type())                           \
      .Output(BACKEND(backend),                                                \
              DATALAYOUT(layout),                                              \
              ::pt::CppTypeToDataType<dtype>::Type());                         \
  PT_TOUCH_KERNEL_REGISTRAR(op_name, backend, layout, dtype)

#define PT_REGISTER_KERNEL_2T(                                             \
    op_name, backend, layout, meta_kernel_fn, dtype1, dtype2)              \
  PT_REGISTER_KERNEL_1T(op_name, backend, layout, meta_kernel_fn, dtype1); \
  PT_REGISTER_KERNEL_1T(op_name, backend, layout, meta_kernel_fn, dtype2)

#define PT_REGISTER_KERNEL_3T(                                        \
    op_name, backend, layout, meta_kernel_fn, dtype1, dtype2, dtype3) \
  PT_REGISTER_KERNEL_2T(                                              \
      op_name, backend, layout, meta_kernel_fn, dtype1, dtype2);      \
  PT_REGISTER_KERNEL_1T(op_name, backend, layout, meta_kernel_fn, dtype3)

#define PT_REGISTER_KERNEL_4T(                                                \
    op_name, backend, layout, meta_kernel_fn, dtype1, dtype2, dtype3, dtype4) \
  PT_REGISTER_KERNEL_2T(                                                      \
      op_name, backend, layout, meta_kernel_fn, dtype1, dtype2);              \
  PT_REGISTER_KERNEL_2T(                                                      \
      op_name, backend, layout, meta_kernel_fn, dtype3, dtype4)

#define PT_REGISTER_KERNEL_5T(op_name,                                   \
                              backend,                                   \
                              layout,                                    \
                              meta_kernel_fn,                            \
                              dtype1,                                    \
                              dtype2,                                    \
                              dtype3,                                    \
                              dtype4,                                    \
                              dtype5)                                    \
  PT_REGISTER_KERNEL_3T(                                                 \
      op_name, backend, layout, meta_kernel_fn, dtype1, dtype2, dtype3); \
  PT_REGISTER_KERNEL_2T(                                                 \
      op_name, backend, layout, meta_kernel_fn, dtype4, dtype5)

#define PT_REGISTER_KERNEL_6T(op_name,                                   \
                              backend,                                   \
                              layout,                                    \
                              meta_kernel_fn,                            \
                              dtype1,                                    \
                              dtype2,                                    \
                              dtype3,                                    \
                              dtype4,                                    \
                              dtype5,                                    \
                              dtype6)                                    \
  PT_REGISTER_KERNEL_3T(                                                 \
      op_name, backend, layout, meta_kernel_fn, dtype1, dtype2, dtype3); \
  PT_REGISTER_KERNEL_3T(                                                 \
      op_name, backend, layout, meta_kernel_fn, dtype4, dtype5, dtype6)

#define PT_REGISTER_KERNEL_7T(op_name,        \
                              backend,        \
                              layout,         \
                              meta_kernel_fn, \
                              dtype1,         \
                              dtype2,         \
                              dtype3,         \
                              dtype4,         \
                              dtype5,         \
                              dtype6,         \
                              ftype7)         \
  PT_REGISTER_KERNEL_4T(op_name,              \
                        backend,              \
                        layout,               \
                        meta_kernel_fn,       \
                        dtype1,               \
                        dtype2,               \
                        dtype3,               \
                        dtype4);              \
  PT_REGISTER_KERNEL_3T(                      \
      op_name, backend, layout, meta_kernel_fn, dtype5, dtype6, dtype7)

#define PT_REGISTER_KERNEL_8T(op_name,        \
                              backend,        \
                              layout,         \
                              meta_kernel_fn, \
                              dtype1,         \
                              dtype2,         \
                              dtype3,         \
                              dtype4,         \
                              dtype5,         \
                              dtype6,         \
                              dtype7,         \
                              dtype8)         \
  PT_REGISTER_KERNEL_4T(op_name,              \
                        backend,              \
                        layout,               \
                        meta_kernel_fn,       \
                        dtype1,               \
                        dtype2,               \
                        dtype3,               \
                        dtype4);              \
  PT_REGISTER_KERNEL_4T(op_name,              \
                        backend,              \
                        layout,               \
                        meta_kernel_fn,       \
                        dtype5,               \
                        dtype6,               \
                        dtype7,               \
                        dtype8)

/**
 * Op Kernel declare macros
 */

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

#define PT_DECLARE_KERNEL_1T(op_name, backend, layout, dtype)                 \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      __dec_pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__,      \
      "PT_DECLARE_KERNEL_*T must be called in global namespace.");            \
  extern int                                                                  \
      TouchOpKernelRegistrar_##op_name##_##backend##_##dtype##_##layout();    \
  UNUSED static int                                                           \
      __declare_pt_op_kernel_##op_name##_##backend##_##layout##_##dtype##__ = \
          TouchOpKernelRegistrar_##op_name##_##backend##_##dtype##_##layout()

#define PT_DECLARE_KERNEL_2T(op_name, backend, layout, dtype1, dtype2) \
  PT_DECLARE_KERNEL_1T(op_name, backend, layout, dtype1);              \
  PT_DECLARE_KERNEL_1T(op_name, backend, layout, dtype2)

#define PT_DECLARE_KERNEL_3T(op_name, backend, layout, dtype1, dtype2, dtype3) \
  PT_REGISTER_KERNEL_2T(op_name, backend, layout, dtype1, dtype2);             \
  PT_DECLARE_KERNEL_1T(op_name, backend, layout, dtype3)

#define PT_DECLARE_KERNEL_4T(                                     \
    op_name, backend, layout, dtype1, dtype2, dtype3, dtype4)     \
  PT_DECLARE_KERNEL_2T(op_name, backend, layout, dtype1, dtype2); \
  PT_DECLARE_KERNEL_2T(op_name, backend, layout, dtype3, dtype4)

#define PT_DECLARE_KERNEL_5T(                                             \
    op_name, backend, layout, dtype1, dtype2, dtype3, dtype4, dtype5)     \
  PT_DECLARE_KERNEL_3T(op_name, backend, layout, dtype1, dtype2, dtype3); \
  PT_DECLARE_KERNEL_2T(op_name, backend, layout, dtype4, dtype5)

#define PT_DECLARE_KERNEL_6T(                                                 \
    op_name, backend, layout, dtype1, dtype2, dtype3, dtype4, dtype5, dtype6) \
  PT_DECLARE_KERNEL_3T(op_name, backend, layout, dtype1, dtype2, dtype3);     \
  PT_DECLARE_KERNEL_3T(op_name, backend, layout, dtype4, dtype5, dtype6)

#define PT_DECLARE_KERNEL_7T(op_name,                            \
                             backend,                            \
                             layout,                             \
                             dtype1,                             \
                             dtype2,                             \
                             dtype3,                             \
                             dtype4,                             \
                             dtype5,                             \
                             dtype6,                             \
                             ftype7)                             \
  PT_DECLARE_KERNEL_4T(                                          \
      op_name, backend, layout, dtype1, dtype2, dtype3, dtype4); \
  PT_DECLARE_KERNEL_3T(op_name, backend, layout, dtype5, dtype6, dtype7)

#define PT_DECLARE_KERNEL_8T(op_name,                            \
                             backend,                            \
                             layout,                             \
                             dtype1,                             \
                             dtype2,                             \
                             dtype3,                             \
                             dtype4,                             \
                             dtype5,                             \
                             dtype6,                             \
                             dtype7,                             \
                             dtype8)                             \
  PT_DECLARE_KERNEL_4T(                                          \
      op_name, backend, layout, dtype1, dtype2, dtype3, dtype4); \
  PT_DECLARE_KERNEL_4T(op_name, backend, layout, dtype5, dtype6, dtype7, dtype8)

}  // namespace pt
