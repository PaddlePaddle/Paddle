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
#include "paddle/framework/attribute.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_info.h"
#include "paddle/framework/op_proto_maker.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

class OpRegistry {
 public:
  template <typename OpType, typename ProtoMakerType, typename GradOpType>
  static void RegisterOp(const std::string& op_type,
                         const std::string& grad_op_type) {
    PADDLE_ENFORCE(!OpInfoMap::Instance().Has(op_type),
                   "'%s' is registered more than once.", op_type);
    OpInfo op_info;
    op_info.creator_ = [](
        const std::string& type, const VariableNameMap& inputs,
        const VariableNameMap& outputs, const AttributeMap& attrs) {
      return new OpType(type, inputs, outputs, attrs);
    };
    op_info.grad_op_type_ = grad_op_type;
    if (std::type_index(typeid(ProtoMakerType)) !=
        std::type_index(typeid(NOPMaker))) {
      op_info.proto_ = new OpProto;
      op_info.checker_ = new OpAttrChecker;
      auto maker = ProtoMakerType(op_info.proto_, op_info.checker_);
      maker.Validate();
      op_info.proto_->set_type(op_type);
      PADDLE_ENFORCE(
          op_info.proto_->IsInitialized(),
          "Fail to initialize %s's OpProto, because %s is not initialized",
          op_type, op_info.proto_->InitializationErrorString());
    } else {
      op_info.proto_ = nullptr;
      op_info.checker_ = nullptr;
    }
    OpInfoMap::Instance().Insert(op_type, op_info);
    // register gradient op
    if (!grad_op_type.empty()) {
      RegisterOp<GradOpType, NOPMaker, NOP>(grad_op_type, "");
    }
  }

  static std::unique_ptr<OperatorBase> CreateOp(const std::string& type,
                                                const VariableNameMap& inputs,
                                                const VariableNameMap& outputs,
                                                AttributeMap attrs);

  static std::unique_ptr<OperatorBase> CreateOp(const OpDesc& op_desc);

  static std::unique_ptr<OperatorBase> CreateGradOp(const OperatorBase& op);

  // compile time InferShape
  static void InferShape(const OpDesc& op_desc,
                         std::map<std::string, VarDesc*>& var_descs) {
    //    auto& info = OpInfoMap::Instance().Get(op_desc.type());
    //    auto op = OpRegistry::CreateOp(op_desc);
  }
};

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

template <typename OpType, typename ProtoMakerType, typename GradOpType>
class OpRegistrar : public Registrar {
 public:
  explicit OpRegistrar(const char* op_type) { OpRegistrar(op_type, ""); }
  OpRegistrar(const char* op_type, const char* grad_op_type) {
    OpRegistry::RegisterOp<OpType, ProtoMakerType, GradOpType>(op_type,
                                                               grad_op_type);
  }
};

template <typename PlaceType, typename KernelType>
class OpKernelRegistrar : public Registrar {
 public:
  explicit OpKernelRegistrar(const char* op_type) {
    OperatorWithKernel::OpKernelKey key;
    key.place_ = PlaceType();
    OperatorWithKernel::AllOpKernels()[op_type][key].reset(new KernelType);
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

/**
 * Macro to register Operator.
 */
#define REGISTER_OP(op_type, op_class, op_maker_class, grad_op_type,          \
                    grad_op_class)                                            \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                             \
      __reg_op__##op_type, "REGISTER_OP must be called in global namespace"); \
  class _OpClass_##op_type##_ : public op_class {                             \
   public:                                                                    \
    DEFINE_OP_CLONE_METHOD(_OpClass_##op_type##_);                            \
    DEFINE_OP_CONSTRUCTOR(_OpClass_##op_type##_, op_class);                   \
  };                                                                          \
  class _OpGradClass_##op_type##_ : public grad_op_class {                    \
   public:                                                                    \
    DEFINE_OP_CLONE_METHOD(_OpGradClass_##op_type##_);                        \
    DEFINE_OP_CONSTRUCTOR(_OpGradClass_##op_type##_, grad_op_class);          \
  };                                                                          \
  static ::paddle::framework::OpRegistrar<                                    \
      _OpClass_##op_type##_, op_maker_class, _OpGradClass_##op_type##_>       \
      __op_registrar_##op_type##__(#op_type, #grad_op_type);                  \
  int TouchOpRegistrar_##op_type() {                                          \
    __op_registrar_##op_type##__.Touch();                                     \
    return 0;                                                                 \
  }

#define REGISTER_OP_WITHOUT_GRADIENT(op_type, op_class, op_maker_class) \
  REGISTER_OP(op_type, op_class, op_maker_class, , ::paddle::framework::NOP)

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

#ifdef PADDLE_ONLY_CPU
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

#define USE_OP(op_type)   \
  USE_OP_ITSELF(op_type); \
  USE_OP_KERNEL(op_type)

}  // namespace framework
}  // namespace paddle
