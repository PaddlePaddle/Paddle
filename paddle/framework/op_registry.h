/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <algorithm>
#include <atomic>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"  // For VLOG()
#include "paddle/framework/attribute.h"
#include "paddle/framework/details/op_registry.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/grad_op_desc_maker.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/shape_inference.h"

namespace paddle {
namespace framework {
class Registrar {
 public:
  // In our design, various kinds of classes, e.g., operators and kernels,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which,
  // however, are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_OP macros to
  // call this method. So, as long as the callee code calls USE_OP, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename... ARGS>
struct OperatorRegistrar : public Registrar {
  explicit OperatorRegistrar(const char* op_type) {
    PADDLE_ENFORCE(!OpInfoMap::Instance().Has(op_type),
                   "'%s' is registered more than once.", op_type);
    static_assert(sizeof...(ARGS) != 0,
                  "OperatorRegistrar should be invoked at least by OpClass");
    OpInfo info;
    details::OperatorRegistrarRecursive<0, false, ARGS...>(op_type, &info);
    OpInfoMap::Instance().Insert(op_type, info);
  }
};

class OpRegistry {
 public:
  template <typename OpType, typename ProtoMakerType, typename GradOpType>
  static void RegisterOp(const std::string& op_type,
                         const std::string& grad_op_type) {
    OperatorRegistrar<OpType, ProtoMakerType> reg(op_type.c_str());
    reg.info.grad_op_type_ = grad_op_type;
    // register gradient op
    if (!grad_op_type.empty()) {
      OperatorRegistrar<GradOpType> grad_reg(grad_op_type.c_str());
    }
  }

  static std::unique_ptr<OperatorBase> CreateOp(const std::string& type,
                                                const VariableNameMap& inputs,
                                                const VariableNameMap& outputs,
                                                AttributeMap attrs);

  static std::unique_ptr<OperatorBase> CreateOp(const OpDesc& op_desc);

  static std::unique_ptr<OperatorBase> CreateOp(const OpDescBind& op_desc);
};

template <typename PlaceType, bool at_end, size_t I, typename... KernelType>
struct OpKernelRegistrarFunctor;

template <typename PlaceType, size_t I, typename... KernelTypes>
struct OpKernelRegistrarFunctor<PlaceType, false, I, KernelTypes...> {
  using KERNEL_TYPE =
      typename std::tuple_element<I, std::tuple<KernelTypes...>>::type;

  void operator()(const char* op_type) const {
    using T = typename KERNEL_TYPE::ELEMENT_TYPE;
    OpKernelType key(ToDataType(std::type_index(typeid(T))), PlaceType());
    OperatorWithKernel::AllOpKernels()[op_type][key].reset(new KERNEL_TYPE);

    constexpr auto size = std::tuple_size<std::tuple<KernelTypes...>>::value;
    OpKernelRegistrarFunctor<PlaceType, I + 1 == size, I + 1, KernelTypes...>
        func;
    func(op_type);
  }
};

template <typename PlaceType, size_t I, typename... KernelType>
struct OpKernelRegistrarFunctor<PlaceType, true, I, KernelType...> {
  void operator()(const char* op_type) const {}
};

// User can register many kernel in one place. The data type could be different.
template <typename PlaceType, typename... KernelType>
class OpKernelRegistrar : public Registrar {
 public:
  explicit OpKernelRegistrar(const char* op_type) {
    OpKernelRegistrarFunctor<PlaceType, false, 0, KernelType...> func;
    func(op_type);
  }
};

/**
 * check if MACRO is used in GLOBAL NAMESPACE.
 */
#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_OPERATOR(op_type, op_class, ...)                      \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      __reg_op__##op_type,                                             \
      "REGISTER_OPERATOR must be called in global namespace");         \
  class _OpClass_##op_type##_ : public op_class {                      \
   public:                                                             \
    DEFINE_OP_CLONE_METHOD(_OpClass_##op_type##_);                     \
    DEFINE_OP_CONSTRUCTOR(_OpClass_##op_type##_, op_class);            \
  };                                                                   \
  static ::paddle::framework::OperatorRegistrar<_OpClass_##op_type##_, \
                                                ##__VA_ARGS__>         \
      __op_registrar_##op_type##__(#op_type);                          \
  int TouchOpRegistrar_##op_type() {                                   \
    __op_registrar_##op_type##__.Touch();                              \
    return 0;                                                          \
  }

/**
 * Macro to register Operator.
 */
#define REGISTER_OP(op_type, op_class, op_maker_class, grad_op_type,       \
                    grad_op_class)                                         \
  REGISTER_OPERATOR(grad_op_type, grad_op_class);                          \
  class _GradOpDescMaker_##grad_op_type##_                                 \
      : public ::paddle::framework::DefaultGradOpDescMaker<true> {         \
    using ::paddle::framework::DefaultGradOpDescMaker<                     \
        true>::DefaultGradOpDescMaker;                                     \
                                                                           \
   protected:                                                              \
    virtual std::string GradOpType() const { return #grad_op_type; }       \
  };                                                                       \
  REGISTER_OPERATOR(op_type, op_class, _GradOpDescMaker_##grad_op_type##_, \
                    op_maker_class);

#define REGISTER_OP_WITH_KERNEL(op_type, ...)                         \
  REGISTER_OPERATOR(op_type, ::paddle::framework::OperatorWithKernel, \
                    ##__VA_ARGS__)

#define REGISTER_OP_WITHOUT_GRADIENT(op_type, op_class, op_maker_class) \
  REGISTER_OPERATOR(op_type, op_class, op_maker_class)

/**
 * Macro to register OperatorKernel.
 */
#define REGISTER_OP_KERNEL(op_type, DEVICE_TYPE, place_class, ...)        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      __reg_op_kernel_##op_type##_##DEVICE_TYPE##__,                      \
      "REGISTER_OP_KERNEL must be called in global namespace");           \
  static ::paddle::framework::OpKernelRegistrar<place_class, __VA_ARGS__> \
      __op_kernel_registrar_##op_type##_##DEVICE_TYPE##__(#op_type);      \
  int TouchOpKernelRegistrar_##op_type##_##DEVICE_TYPE() {                \
    __op_kernel_registrar_##op_type##_##DEVICE_TYPE##__.Touch();          \
    return 0;                                                             \
  }

#define REGISTER_OP_GPU_KERNEL(op_type, ...) \
  REGISTER_OP_KERNEL(op_type, GPU, ::paddle::platform::GPUPlace, __VA_ARGS__)

#define REGISTER_OP_CPU_KERNEL(op_type, ...) \
  REGISTER_OP_KERNEL(op_type, CPU, ::paddle::platform::CPUPlace, __VA_ARGS__)

/**
 * Macro to mark what Operator and Kernel
 * we will use and tell the compiler to
 * link them into target.
 */
#define USE_OP_ITSELF(op_type)                                    \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                 \
      __use_op_itself_##op_type,                                  \
      "USE_OP_ITSELF must be called in global namespace");        \
  extern int TouchOpRegistrar_##op_type();                        \
  static int use_op_itself_##op_type##_ __attribute__((unused)) = \
      TouchOpRegistrar_##op_type()

#define USE_OP_DEVICE_KERNEL(op_type, DEVICE_TYPE)               \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                \
      __use_op_kernel_##op_type##_##DEVICE_TYPE##__,             \
      "USE_OP_DEVICE_KERNEL must be in global namespace");       \
  extern int TouchOpKernelRegistrar_##op_type##_##DEVICE_TYPE(); \
  static int use_op_kernel_##op_type##_##DEVICE_TYPE##_          \
      __attribute__((unused)) =                                  \
          TouchOpKernelRegistrar_##op_type##_##DEVICE_TYPE()

// TODO(fengjiayi): The following macros
// seems ugly, do we have better method?

#ifndef PADDLE_WITH_CUDA
#define USE_OP_KERNEL(op_type) USE_OP_DEVICE_KERNEL(op_type, CPU)
#else
#define USE_OP_KERNEL(op_type)        \
  USE_OP_DEVICE_KERNEL(op_type, CPU); \
  USE_OP_DEVICE_KERNEL(op_type, GPU)
#endif

#define USE_NO_KERNEL_OP(op_type) USE_OP_ITSELF(op_type);

#define USE_CPU_ONLY_OP(op_type) \
  USE_OP_ITSELF(op_type);        \
  USE_OP_DEVICE_KERNEL(op_type, CPU);

#define USE_GPU_ONLY_OP(op_type) \
  USE_OP_ITSELF(op_type);        \
  USE_OP_DEVICE_KERNEL(op_type, GPU)

#define USE_OP(op_type)   \
  USE_OP_ITSELF(op_type); \
  USE_OP_KERNEL(op_type)

}  // namespace framework
}  // namespace paddle
