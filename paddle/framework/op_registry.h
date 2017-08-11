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
#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_desc.pb.h"
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
    VarProto* var_;
    std::function<void()> on_multiple_;
    std::function<void()> on_temporary_;

    VariableBuilder& SetMultiple() {
      var_->set_multiple(true);
      on_multiple_();
      return *this;
    }

    VariableBuilder& SetTemporary() {
      PADDLE_ENFORCE(bool(on_temporary_), "Cannot set temporary");
      var_->set_temporary(true);
      on_temporary_();
      return *this;
    }

    VariableBuilder& IgnoreGradient() {
      var_->set_ignore_gradient(true);
      return *this;
    }
  };

  VariableBuilder AddInput(const std::string& name,
                           const std::string& comment) {
    auto input = proto_->mutable_inputs()->Add();
    *input->mutable_name() = name;
    *input->mutable_comment() = comment;
    return VariableBuilder{input, [=] { this->SetHasMultipleInput(); },
                           nullptr};
  }

  VariableBuilder AddOutput(const std::string& name,
                            const std::string& comment) {
    auto output = proto_->mutable_outputs()->Add();
    *output->mutable_name() = name;
    *output->mutable_comment() = comment;
    return VariableBuilder{output, [=] { this->SetHasMultipleOutput(); },
                           [=] { this->SetHasTemporaryOutput(); }};
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
  void SetHasMultiple(const std::string& in_out, bool* flag) {
    if (!*flag) {
      AddAttr<std::vector<int>>(in_out + "_format",
                                "The multiple index of " + in_out +
                                    "\n"
                                    R"DOC(
This attribute is used by Paddle core framework. Paddle's Op support each input
or output could be a list of variable. This attribute is used to show how that
list organized.

e.g.
  input = ["a", "b", "c", "d", "e", "f"]
  input_format = [0, 4, 5, 6]

means
  The number of all input variables this op is six, and they are segmented into
  three inputs.

  The first input is input[0:4], second is input[4:5], third is input[5:6].
)DOC",
                                /*generated*/ true);
      *flag = true;
    }
  }

  void SetHasMultipleInput() { SetHasMultiple("input", &has_multiple_input_); }
  void SetHasMultipleOutput() {
    SetHasMultiple("output", &has_multiple_output_);
  }

  void SetHasTemporaryOutput() {
    if (!has_temporary_output_) {
      AddAttr<std::vector<int>>("temporary_index",
                                R"DOC(The temporary index of output.

Not all output of Paddle Op is used by user. For faster computation, each op
could output some its internal state to other op, other op could take that
output to make compute faster.

Add a mark to which output is temporary is helpful for future optimization.
)DOC",
                                /*generated*/ true)
          .SetDefault(std::vector<int>());
      has_temporary_output_ = true;
    }
  }

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
  bool has_multiple_input_{false};
  bool has_multiple_output_{false};
  bool has_temporary_output_{false};
};

class NOPMaker : public OpProtoAndCheckerMaker {};

struct OpInfo {
  std::function creator_;
  std::string grad_op_type_;
  OpProto* proto_;
  OpAttrChecker* checker_;
};

class OpRegistry {
  using OpCreator = std::function<OperatorBase*()>;
  using VarIndexMap = std::unordered_map<std::string, int>;
  using VarNameList = std::vector<std::string>;

 public:
  template <typename OpType, typename ProtoMakerType>
  static void RegisterOp(const std::string& op_type,
                         const std::string& grad_op_type) {
    PADDLE_ENFORCE(op_info_map().count(op_type) == 0,
                   "'%s' is registered more than once.", op_type);
    OpInfo op_info;
    op_info.creator_ = [] { return new OpType; };
    op_info.grad_op_type_ = grad_op_type;
    if (std::type_index(typeid(ProtoMakerType)) !=
        std::type_index(typeid(NOPMaker))) {
      op_info.proto_ = new OpProto;
      op_info.op_checker_ = new OpAttrChecker;
      auto maker = ProtoMakerType(op_info.proto_, op_info.op_checker_);
      maker.Validate();
      *op_info.proto_->mutable_type() = op_type;
      PADDLE_ENFORCE(
          op_info.proto_->IsInitialized(),
          "Fail to initialize %s's OpProto, because %s is not initialized",
          op_type, op_info.proto_->InitializationErrorString());
      //======will be refactored in following PRs============//
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
      //================================================//
    }
    op_info_map.insert(std::make_pair(op_type, op_info));
  }

  static std::shared_ptr<OperatorBase> CreateOp(const std::string& type,
                                                const VarNameList& inputs,
                                                const VarNameList& outputs,
                                                const AttributeMap& attrs) {
    auto it = op_info_map().find(type);
    PADDLE_ENFORCE(it != op_info_map().end(), "'%s' has not been registered.",
                   type);

    auto op = it->second.creator_();
    op->type_ = type;
    op->inputs_ = inputs;
    op->outputs_ = outputs;

    op->attrs_ = attrs;
    it->second.checker_->Check(op->attrs_);

    GenerateTempVariableName(op);

    {
      auto var_index_it = VarIndexMaps().find(type);
      if (var_index_it != VarIndexMaps().end()) {
        op->in_out_idxs_ = var_index_it->second;
      }
    }

    op->Init();
    return std::shared_ptr<OperatorBase>(op);
  }

  static std::shared_ptr<OperatorBase> CreateOp(const OpDesc& op_desc) {
    std::vector<std::string> inputs;
    inputs.reserve((size_t)op_desc.inputs_size());
    std::copy(op_desc.inputs().begin(), op_desc.inputs().end(),
              std::back_inserter(inputs));

    std::vector<std::string> outputs;
    outputs.reserve((size_t)op_desc.outputs_size());
    std::copy(op_desc.outputs().begin(), op_desc.outputs().end(),
              std::back_inserter(outputs));

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

  static std::unordered_map<const std::string, const OpInfo>& op_info_map() {
    static std::unordered_map<const std::string, const OpInfo> op_info_map_;
    return op_info_map_;
  }

  static std::unordered_map<std::string, std::shared_ptr<VarIndexMap>>&
  VarIndexMaps() {
    static std::unordered_map<std::string, std::shared_ptr<VarIndexMap>> maps_;
    return maps_;
  }

 private:
  static void GenerateTempVariableName(OperatorBase* op) {
    static std::atomic<size_t> gUniqId(0UL);
    for (auto& outname : op->outputs_) {
      if (outname == kTempVarName) {
        outname += op->type_;
        outname += "@";
        outname += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
};

class Registrar {
 public:
  // In our design, various kinds of classes, e.g., operators and kernels, have
  // their corresponding registry and registrar. The action of registration is
  // in the constructor of a global registrar variable, which, however, are not
  // used in the code that calls package framework, and would be removed from
  // the generated binary file by the linker. To avoid such removal, we add
  // Touch to all registrar classes and make USE_OP macros to call this
  // method. So, as long as the callee code calls USE_OP, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename OpType, typename ProtoMakerType>
class OpRegistrar : public Registrar {
 public:
  OpRegistrar(const char* op_type) { OpRegistrar(op_type, ""); }
  OpRegistrar(const char* op_type, const char* grad_op_type) {
    OpRegistry::RegisterOp<OpType, ProtoMakerType>(op_type, grad_op_type);
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
#define REGISTER_OP(op_type, op_class, op_maker_class, grad_op_type)          \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                             \
      __reg_op__##op_type, "REGISTER_OP must be called in global namespace"); \
  static ::paddle::framework::OpRegistrar<op_class, op_maker_class>           \
      __op_registrar_##op_type##__(#op_type, #grad_op_type);                  \
  int TouchOpRegistrar_##op_type() {                                          \
    __op_registrar_##op_type##__.Touch();                                     \
    return 0;                                                                 \
  }

#define REGISTER_OP_WITHOUT_GRADIENT(op_type, op_class, op_maker_class) \
  REGISTER_OP(op_type, op_class, op_maker_class, )

#define REGISTER_GRADIENT_OP(op_type, op_class) \
  REGISTER_OP(op_type, op_class, ::paddle::framework::NOPMaker, )

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

/**
 * Macro to Forbid user register Gradient Operator.
 */
/*
#define NO_GRADIENT(op_type)                           \
 STATIC_ASSERT_GLOBAL_NAMESPACE(                      \
     __reg_gradient_op__##op_type##_##op_type##_grad, \
     "NO_GRADIENT must be called in global namespace")
*/

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
