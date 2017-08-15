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
#include "paddle/framework/operator.h"
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

    VariableBuilder& AsDuplicable() {
      var_->set_duplicable(true);
      return *this;
    }

    VariableBuilder& AsIntermediate() {
      var_->set_intermediate(true);
      return *this;
    }

    // TODO(FengJiayi, yuyang18): `AsNoGradient` is a very bad name, because it
    // means that input/output is not needed when calculate gradient. It does
    // not mean no gradient when backward. It should be changed soon.
    VariableBuilder& AsNoGradient() {
      var_->set_no_gradient(true);
      return *this;
    }
  };

  VariableBuilder AddInput(const std::string& name,
                           const std::string& comment) {
    auto* input = proto_->add_inputs();
    input->set_name(name);
    input->set_comment(comment);
    return VariableBuilder{input};
  }

  VariableBuilder AddOutput(const std::string& name,
                            const std::string& comment) {
    auto* output = proto_->add_outputs();
    output->set_name(name);
    output->set_comment(comment);
    return VariableBuilder{output};
  }

  template <typename T>
  TypedAttrChecker<T>& AddAttr(const std::string& name,
                               const std::string& comment,
                               bool generated = false) {
    auto* attr = proto_->add_attrs();
    attr->set_name(name);
    attr->set_comment(comment);
    attr->set_generated(generated);
    attr->set_type(AttrTypeID<T>());
    return op_checker_->AddAttrChecker<T>(name);
  }

  void AddComment(const std::string& comment) { proto_->set_comment(comment); }

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

class NOPMaker : public OpProtoAndCheckerMaker {
 public:
  NOPMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {}
};

class OpRegistry {
  using VarNameMap = OperatorBase::VarNameMap;
  using OpCreator = std::function<OperatorBase*(
      const std::string& /*type*/, const VarNameMap& /*inputs*/,
      const VarNameMap& /*outputs*/, const AttributeMap& /*attrs*/)>;

 public:
  struct OpInfo {
    OpCreator creator_;
    std::string grad_op_type_;
    OpProto* proto_;
    OpAttrChecker* checker_;
  };

  template <typename OpType, typename ProtoMakerType, typename GradOpType>
  static void RegisterOp(const std::string& op_type,
                         const std::string& grad_op_type) {
    PADDLE_ENFORCE(op_info_map().count(op_type) == 0,
                   "'%s' is registered more than once.", op_type);
    OpInfo op_info;
    op_info.creator_ = [](const std::string& type, const VarNameMap& inputs,
                          const VarNameMap& outputs,
                          const AttributeMap& attrs) {
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
    op_info_map().insert(std::make_pair(op_type, op_info));
    // register gradient op
    if (!grad_op_type.empty()) {
      RegisterOp<GradOpType, NOPMaker, NOP>(grad_op_type, "");
    }
  }

  static std::shared_ptr<OperatorBase> CreateOp(const std::string& type,
                                                const VarNameMap& inputs,
                                                const VarNameMap& outputs,
                                                AttributeMap attrs) {
    auto it = op_info_map().find(type);
    PADDLE_ENFORCE(it != op_info_map().end(),
                   "Operator '%s' has not been registered.", type);
    it->second.checker_->Check(attrs);
    auto op = it->second.creator_(type, inputs, outputs, attrs);
    return std::shared_ptr<OperatorBase>(op);
  }

  static VarNameMap ConvertOpDescVarsToVarNameMap(
      const google::protobuf::RepeatedPtrField<OpDesc::Var>& op_desc_vars) {
    VarNameMap ret_val;
    for (auto& var : op_desc_vars) {
      auto& var_names = ret_val[var.parameter()];
      auto& var_names_in_proto = var.arguments();
      var_names.reserve(static_cast<size_t>(var_names_in_proto.size()));
      std::copy(var_names_in_proto.begin(), var_names_in_proto.end(),
                std::back_inserter(var_names));
    }
    return ret_val;
  }

  static std::shared_ptr<OperatorBase> CreateOp(const OpDesc& op_desc) {
    VarNameMap inputs = ConvertOpDescVarsToVarNameMap(op_desc.inputs());
    VarNameMap outputs = ConvertOpDescVarsToVarNameMap(op_desc.outputs());
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
    return grad_op;
  }

  static std::unordered_map<std::string, const OpInfo>& op_info_map() {
    static std::unordered_map<std::string, const OpInfo> op_info_map_;
    return op_info_map_;
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
  static ::paddle::framework::OpRegistrar<op_class, op_maker_class,           \
                                          grad_op_class>                      \
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
 * Macro to mark what Operator and Kernel we will use and tell the compiler to
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

// TODO(fengjiayi): The following macros seems ugly, do we have better method?

#ifdef PADDLE_ONLY_CPU
#define USE_OP_KERNEL(op_type) USE_OP_DEVICE_KERNEL(op_type, CPU)
#else
#define USE_OP_KERNEL(op_type)        \
  USE_OP_DEVICE_KERNEL(op_type, CPU); \
  USE_OP_DEVICE_KERNEL(op_type, GPU)
#endif

#define USE_CPU_ONLY_OP(op_type) \
  USE_OP_ITSELF(op_type);        \
  USE_OP_DEVICE_KERNEL(op_type, CPU);

#define USE_OP(op_type)   \
  USE_OP_ITSELF(op_type); \
  USE_OP_KERNEL(op_type)

}  // namespace framework
}  // namespace paddle
