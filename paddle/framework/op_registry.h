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
#include <unordered_map>
#include <unordered_set>
#include "paddle/framework/attribute.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

// this class not only make proto but also init attribute checkers.
class OpProtoAndCheckerMaker {
 public:
  OpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : proto_(proto), op_checker_(op_checker) {}

  ~OpProtoAndCheckerMaker() {
    PADDLE_ENFORCE(validated_, "should call Validate after build");
  }

  void Validate() {
    validated_ = true;
    CheckNoDuplicatedInOutAttrs();
  }

 protected:
  struct VariableBuilder {
    OpProto::Var* var_;

    VariableBuilder& SetMultiple() {
      var_->set_duplicable(true);
      return *this;
    }

    VariableBuilder& SetTemporary() {
      var_->set_intermediate(true);
      return *this;
    }

    VariableBuilder& IgnoreGradient() {
      var_->set_no_gradient(true);
      return *this;
    }
  };

  VariableBuilder AddInput(const std::string& name,
                           const std::string& comment) {
    auto input = proto_->mutable_inputs()->Add();
    *input->mutable_name() = name;
    *input->mutable_comment() = comment;
    return VariableBuilder{input};
  }

  VariableBuilder AddOutput(const std::string& name,
                            const std::string& comment) {
    auto output = proto_->mutable_outputs()->Add();
    *output->mutable_name() = name;
    *output->mutable_comment() = comment;
    return VariableBuilder{output};
  }

  template <typename T>
  TypedAttrChecker<T>& AddAttr(const std::string& name,
                               const std::string& comment,
                               bool generated = false) {
    auto attr = proto_->mutable_attrs()->Add();
    *attr->mutable_name() = name;
    *attr->mutable_comment() = comment;
    attr->set_generated(generated);
    attr->set_type(AttrTypeID<T>());
    return op_checker_->AddAttrChecker<T>(name);
  }

  void AddComment(const std::string& comment) {
    *(proto_->mutable_comment()) = comment;
  }

 private:
  void CheckNoDuplicatedInOutAttrs() {
    std::unordered_set<std::string> names;
    auto checker = [&](const std::string& name) {
      PADDLE_ENFORCE(!names.count(name), "[%s] is duplicated", name);
      names.insert(name);
    };
    for (auto& attr : proto_->attrs()) {
      checker(attr.name());
    }
    for (auto& input : proto_->inputs()) {
      checker(input.name());
    }
    for (auto& output : proto_->outputs()) {
      checker(output.name());
    }
  }

  OpProto* proto_;
  OpAttrChecker* op_checker_;
  bool validated_{false};
};

class OpRegistry {
  using OpCreator = std::function<OperatorBase*()>;
  using VarIndexMap = std::unordered_map<std::string, int>;
  using VarNameMap = std::unordered_map<std::string, std::vector<std::string>>;

 public:
  template <typename OpType, typename ProtoMakerType>
  static void RegisterOp(const std::string& op_type) {
    op_creators()[op_type] = [] { return new OpType; };
    OpAttrChecker& op_checker = op_checkers()[op_type];
    OpProto& op_proto = protos()[op_type];
    auto maker = ProtoMakerType(&op_proto, &op_checker);
    maker.Validate();
    *op_proto.mutable_type() = op_type;
    PADDLE_ENFORCE(
        op_proto.IsInitialized(),
        "Fail to initialize %s's OpProto, because %s is not initialized",
        op_type, op_proto.InitializationErrorString());

    VarIndexMaps()[op_type].reset(new VarIndexMap());
    auto& varmap = *VarIndexMaps()[op_type];
    int idx = 0;
    for (auto& var : op_proto.inputs()) {
      varmap[var.name()] = idx++;
    }
    idx = 0;
    for (auto& var : op_proto.outputs()) {
      varmap[var.name()] = idx++;
    }
  }

  template <typename GradOpType>
  static void RegisterGradOp(const std::string& op_type,
                             const std::string& grad_op_type) {
    op_creators()[grad_op_type] = [] { return new GradOpType; };
    grad_ops()[op_type] = grad_op_type;
  }

  static std::shared_ptr<OperatorBase> CreateOp(const std::string& type,
                                                const VarNameMap& inputs,
                                                const VarNameMap& outputs,
                                                const AttributeMap& attrs) {
    auto op_create_it = op_creators().find(type);
    PADDLE_ENFORCE(op_create_it != op_creators().end(),
                   "Operator %s cannot be found.", type);

    auto op = op_create_it->second();
    op->type_ = type;
    op->inputs_ = inputs;
    op->outputs_ = outputs;

    op->attrs_ = attrs;
    op_checkers().at(type).Check(op->attrs_);

    GenerateTempVariableName(op);

    op->Init();
    return std::shared_ptr<OperatorBase>(op);
  }

  static std::shared_ptr<OperatorBase> CreateOp(const OpDesc& op_desc) {
    VarNameMap inputs;
    for (auto& input : op_desc.inputs()) {
      auto& var_names = inputs[input.op_proto_name()];
      auto& var_names_in_proto = input.var_names();
      var_names.reserve(static_cast<size_t>(var_names_in_proto.size()));
      std::copy(var_names_in_proto.begin(), var_names_in_proto.end(),
                std::back_inserter(var_names));
    }

    VarNameMap outputs;
    for (auto& output : op_desc.outputs()) {
      auto& var_names = outputs[output.op_proto_name()];
      auto& var_names_in_proto = output.var_names();
      var_names.reserve(static_cast<size_t>(var_names_in_proto.size()));
      std::copy(var_names_in_proto.begin(), var_names_in_proto.end(),
                std::back_inserter(var_names));
    }

    AttributeMap attrs;
    for (auto& attr : op_desc.attrs()) {
      attrs[attr.name()] = GetAttrValue(attr);
    }

    return CreateOp(op_desc.type(), inputs, outputs, attrs);
  }

  static std::shared_ptr<OperatorBase> CreateGradOp(const OperatorBase& op) {
    PADDLE_ENFORCE(!op.IsNetOp(),
                   "Use framework::Backward to get backward ops");
    std::shared_ptr<OperatorBase> grad_op(BuildGradOp(&op));
    grad_op->Init();
    return grad_op;
  }

  static std::unordered_map<std::string, OpProto>& protos() {
    static std::unordered_map<std::string, OpProto> protos_;
    return protos_;
  }

  static std::unordered_map<std::string, std::string>& grad_ops() {
    static std::unordered_map<std::string, std::string> grad_ops_;
    return grad_ops_;
  }

  static std::unordered_map<std::string, std::shared_ptr<VarIndexMap>>&
  VarIndexMaps() {
    static std::unordered_map<std::string, std::shared_ptr<VarIndexMap>> maps_;
    return maps_;
  }

  static std::unordered_map<std::string, OpCreator>& op_creators() {
    static std::unordered_map<std::string, OpCreator> op_creators_;
    return op_creators_;
  }

 private:
  static std::unordered_map<std::string, OpAttrChecker>& op_checkers() {
    static std::unordered_map<std::string, OpAttrChecker> op_checkers_;
    return op_checkers_;
  }

  static void GenerateTempVariableName(OperatorBase* op) {
    static std::atomic<size_t> gUniqId(0UL);
    for (auto& output : op->outputs_) {
      for (auto& output_name : output.second) {
        if (output_name == kTempVarName) {
          output_name += op->type_;
          output_name += "@";
          output_name += std::to_string(gUniqId.fetch_add(1));
        }
      }
    }
  }
};

template <typename OpType, typename ProtoMakerType>
class OpRegisterHelper {
 public:
  explicit OpRegisterHelper(const char* op_type) {
    OpRegistry::RegisterOp<OpType, ProtoMakerType>(op_type);
  }
};

template <typename GradOpType>
class GradOpRegisterHelper {
 public:
  GradOpRegisterHelper(const char* op_type, const char* grad_op_type) {
    OpRegistry::RegisterGradOp<GradOpType>(op_type, grad_op_type);
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
 * Macro to Register Operator.
 */
#define REGISTER_OP(__op_type, __op_class, __op_maker_class)                 \
  STATIC_ASSERT_GLOBAL_NAMESPACE(__reg_op__##__op_type,                      \
                                 "REGISTER_OP must be in global namespace"); \
  static ::paddle::framework::OpRegisterHelper<__op_class, __op_maker_class> \
      __op_register_##__op_type##__(#__op_type);                             \
  int __op_register_##__op_type##_handle__() { return 0; }

/**
 * Macro to Register Gradient Operator.
 */
#define REGISTER_GRADIENT_OP(__op_type, __grad_op_type, __grad_op_class)       \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                              \
      __reg_gradient_op__##__op_type##__grad_op_type,                          \
      "REGISTER_GRADIENT_OP must be in global namespace");                     \
  static ::paddle::framework::GradOpRegisterHelper<__grad_op_class>            \
      __op_gradient_register_##__op_type##__grad_op_type##__(#__op_type,       \
                                                             #__grad_op_type); \
  int __op_gradient_register_##__op_type##__grad_op_type##_handle__() {        \
    return 0;                                                                  \
  }

/**
 * Macro to Forbid user register Gradient Operator.
 */
#define NO_GRADIENT(__op_type)                          \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                       \
      __reg_gradient_op__##__op_type##__op_type##_grad, \
      "NO_GRADIENT must be in global namespace")

/**
 * Macro to Register OperatorKernel.
 */
#define REGISTER_OP_KERNEL(type, DEVICE_TYPE, PlaceType, ...)             \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      __reg_op_kernel_##type##_##DEVICE_TYPE##__,                         \
      "REGISTER_OP_KERNEL must be in global namespace");                  \
  struct __op_kernel_register__##type##__##DEVICE_TYPE##__ {              \
    __op_kernel_register__##type##__##DEVICE_TYPE##__() {                 \
      ::paddle::framework::OperatorWithKernel::OpKernelKey key;           \
      key.place_ = PlaceType();                                           \
      ::paddle::framework::OperatorWithKernel::AllOpKernels()[#type][key] \
          .reset(new __VA_ARGS__());                                      \
    }                                                                     \
  };                                                                      \
  static __op_kernel_register__##type##__##DEVICE_TYPE##__                \
      __reg_kernel_##type##__##DEVICE_TYPE##__;                           \
  int __op_kernel_register_##type##_handle_##DEVICE_TYPE##__() { return 0; }

// (type, KernelType)
#define REGISTER_OP_GPU_KERNEL(type, ...) \
  REGISTER_OP_KERNEL(type, GPU, ::paddle::platform::GPUPlace, __VA_ARGS__)

// (type, KernelType)
#define REGISTER_OP_CPU_KERNEL(type, ...) \
  REGISTER_OP_KERNEL(type, CPU, ::paddle::platform::CPUPlace, __VA_ARGS__)

/**
 * Macro to mark what Operator and Kernel we will use and tell the compiler to
 * link them into target.
 */
#define USE_OP_WITHOUT_KERNEL(op_type)                      \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                           \
      __use_op_without_kernel_##op_type,                    \
      "USE_OP_WITHOUT_KERNEL must be in global namespace"); \
  extern int __op_register_##op_type##_handle__();          \
  static int __use_op_ptr_##op_type##_without_kernel__      \
      __attribute__((unused)) = __op_register_##op_type##_handle__()

#define USE_OP_KERNEL(op_type, DEVICE_TYPE)                               \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      __use_op_kernel_##op_type##_##DEVICE_TYPE##__,                      \
      "USE_OP_KERNEL must be in global namespace");                       \
  extern int __op_kernel_register_##op_type##_handle_##DEVICE_TYPE##__(); \
  static int __use_op_ptr_##op_type##_##DEVICE_TYPE##_kernel__            \
      __attribute__((unused)) =                                           \
          __op_kernel_register_##op_type##_handle_##DEVICE_TYPE##__()

// use Operator with only cpu kernel.
#define USE_OP_CPU(op_type)       \
  USE_OP_WITHOUT_KERNEL(op_type); \
  USE_OP_KERNEL(op_type, CPU)

#ifdef PADDLE_ONLY_CPU
#define USE_OP(op_type) USE_OP_CPU(op_type)
#else
#define USE_OP(op_type) \
  USE_OP_CPU(op_type);  \
  USE_OP_KERNEL(op_type, GPU)
#endif

}  // namespace framework
}  // namespace paddle
