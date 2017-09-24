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
#include <string>
#include <unordered_map>
#include <vector>

#include "op_info.h"
#include "paddle/framework/attribute.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/shape_inference.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/place.h"
#include "paddle/platform/variant.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace framework {

/// If a variable is a empty variable, that name will be used.
constexpr char kEmptyVarName[] = "@EMPTY@";

/// If a variable is a temporary variable, that name will be set in Python,
/// but it will be convert to a unique name in scope after OpCreator.
constexpr char kTempVarName[] = "@TEMP@";

/// If a variable's name has a certain suffix, it means that the
/// variable is the gradient of another varibale.
/// e.g. Variable "x@GRAD" is the gradient of varibale "x".
constexpr char kGradVarSuffix[] = "@GRAD";

/// Variables with this suffix are supposed to be filled up with zeros.
constexpr char kZeroVarSuffix[] = "@ZERO";

inline std::string GradVarName(const std::string& var_name) {
  return var_name + kGradVarSuffix;
}

class OperatorBase;
class InferShapeContext;
class ExecutionContext;

extern const Tensor* GetTensorFromVar(const Variable* var);

/**
 * OperatorBase has the basic element that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class OperatorBase {
 public:
  OperatorBase(const std::string& type, const VariableNameMap& inputs,
               const VariableNameMap& outputs, const AttributeMap& attrs);

  virtual ~OperatorBase() {}

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }

  virtual std::string DebugString() const;

  /// InferShape infer the size of Variables used by this Operator with
  /// information inside scope
  virtual void InferShape(const Scope& scope) const = 0;

  /// Net will call this function to Run an op.
  virtual void Run(const Scope& scope,
                   const platform::DeviceContext& dev_ctx) const = 0;

  virtual bool IsNetOp() const { return false; }

  virtual bool SupportGPU() const { return false; }

  /// rename inputs outputs name
  void Rename(const std::string& old_name, const std::string& new_name);

  const VariableNameMap& Inputs() const { return inputs_; }
  const VariableNameMap& Outputs() const { return outputs_; }

  //! Get a input with argument's name described in `op_proto`
  std::string Input(const std::string& name) const;
  //! Get a input which has multiple variables.
  const std::vector<std::string>& Inputs(const std::string& name) const;

  std::vector<std::string> InputVars() const;

  //! Get a output with argument's name described in `op_proto`
  std::string Output(const std::string& name) const;
  //! Get an output which has multiple variables.
  //! TODO add a vector_view to prevent memory copy.
  const std::vector<std::string>& Outputs(const std::string& name) const;

  virtual std::vector<std::string> OutputVars(bool has_intermediate) const;

  const std::string& Type() const { return type_; }
  void SetType(const std::string& type) { type_ = type; }
  const AttributeMap& Attrs() const { return attrs_; }

  // Return a new operator instance, which is as same as this.
  // Use unique_ptr to prevent caller forget to delete this pointer.
  virtual std::unique_ptr<OperatorBase> Clone() const = 0;

 protected:
  std::string type_;
  // NOTE: in case of OpGrad, inputs_ contains:
  // I (Inputs)opear
  // O (Outputs)
  // OG (Output Gradients)
  VariableNameMap inputs_;

  // NOTE: in case of OpGrad, outputs_ contains
  // IG (Inputs Gradients)
  VariableNameMap outputs_;
  AttributeMap attrs_;

 private:
  void GenerateTemporaryNames();
  void CheckAllInputOutputSet() const;
};

// Macro for define a clone method.
// If you are writing an kernel operator, `Clone` will be defined when you
// register it. i.e. `Clone` method is not needed to define by yourself.
#define DEFINE_OP_CLONE_METHOD(cls)                       \
  std::unique_ptr<OperatorBase> Clone() const final {     \
    return std::unique_ptr<OperatorBase>(new cls(*this)); \
  }

// Macro for define a default constructor for Operator.
// You can also use
//   using PARENT_CLASS::PARENT_CLASS;
// to use parent's constructor.
#define DEFINE_OP_CONSTRUCTOR(cls, parent_cls)             \
  cls(const std::string& type,                             \
      const ::paddle::framework::VariableNameMap& inputs,  \
      const ::paddle::framework::VariableNameMap& outputs, \
      const paddle::framework::AttributeMap& attrs)        \
      : parent_cls(type, inputs, outputs, attrs) {}

class NOP : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  void InferShape(const Scope& scope) const override {}
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {}
  std::unique_ptr<OperatorBase> Clone() const override {
    return std::unique_ptr<OperatorBase>(new NOP(*this));
  }
};

class InferShapeContext {
 public:
  InferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  const OperatorBase& op() const { return op_; }

  const Scope& scope() const { return scope_; }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return op_.Attr<T>(name);
  }

  size_t InputSize(const std::string& name) const {
    return op_.Inputs(name).size();
  }

  size_t OutputSize(const std::string& name) const {
    return op_.Outputs(name).size();
  }

  const Variable* InputVar(const std::string& name) const {
    auto ipt = op_.Input(name);
    return ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
  }

  Variable* OutputVar(const std::string& name) const {
    auto opt = op_.Output(name);
    return opt == kEmptyVarName ? nullptr : scope_.FindVar(opt);
  }

  const std::vector<const Variable*> MultiInputVar(
      const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const Variable*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [this](const std::string& name) {
                     return name == kEmptyVarName ? nullptr
                                                  : scope_.FindVar(name);
                   });
    return res;
  }

  std::vector<const Variable*> MultiOutputVar(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<const Variable*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [this](const std::string& name) {
                     return name == kEmptyVarName ? nullptr
                                                  : scope_.FindVar(name);
                   });
    return res;
  }

  template <typename T>
  const T* Input(const std::string& name) const {
    auto* var = InputVar(name);
    return var == nullptr ? nullptr : &var->Get<T>();
  }

  template <typename T>
  T* Output(const std::string& name) const {
    auto var = OutputVar(name);
    return var == nullptr ? nullptr : var->GetMutable<T>();
  }

  template <typename T>
  const std::vector<const T*> MultiInput(const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const T*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [&](const std::string& sub_name) {
                     auto var = scope_.FindVar(sub_name);
                     return var == nullptr ? nullptr : &var->Get<T>();
                   });
    return res;
  }

  template <typename T>
  std::vector<T*> MultiOutput(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<T*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [&](const std::string& sub_name) {
                     auto var = scope_.FindVar(sub_name);
                     return var == nullptr ? nullptr : var->GetMutable<T>();
                   });
    return res;
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
};

template <>
const Tensor* InferShapeContext::Input<Tensor>(const std::string& name) const;

template <>
const std::vector<const Tensor*> InferShapeContext::MultiInput<Tensor>(
    const std::string& name) const;

template <typename T>
struct EigenDeviceConverter;

template <>
struct EigenDeviceConverter<platform::CPUPlace> {
  using EigenDeviceType = Eigen::DefaultDevice;
};

#ifndef PADDLE_ONLY_CPU
template <>
struct EigenDeviceConverter<platform::GPUPlace> {
  using EigenDeviceType = Eigen::GpuDevice;
};
#endif

class ExecutionContext : public InferShapeContext {
 public:
  ExecutionContext(const OperatorBase& op, const Scope& scope,
                   const platform::DeviceContext& device_context)
      : InferShapeContext(op, scope), device_context_(device_context) {}

  template <typename PlaceType,
            typename DeviceType =
                typename EigenDeviceConverter<PlaceType>::EigenDeviceType>
  DeviceType& GetEigenDevice() const;

  platform::Place GetPlace() const { return device_context_.GetPlace(); }

  const platform::DeviceContext& device_context() const {
    return device_context_;
  }

  // redefine Output function,
  // use Variable::Get instead of Variable::GetMutable
  template <typename T>
  T* Output(const std::string& name) const {
    auto var = OutputVar(name);
    return var == nullptr ? nullptr : const_cast<T*>(&var->Get<T>());
  }

  // redefine MultiOutput function.
  // use Variable::Get instead of Variable::GetMutable
  template <typename T>
  std::vector<T*> MultiOutput(const std::string& name) const {
    auto names = op().Outputs(name);
    std::vector<T*> res;
    res.reserve(names.size());
    std::transform(
        names.begin(), names.end(), std::back_inserter(res),
        [&](const std::string& sub_name) { return Output<T>(sub_name); });
    return res;
  }

 private:
  const platform::DeviceContext& device_context_;
};

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const;

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const;

class BlockDesc {
 public:
  explicit BlockDesc(const std::map<std::string, VarDesc*>& var_descs)
      : var_descs_(var_descs) {}
  ~BlockDesc() {}

  VarDesc* GetVar(const std::string& name) const {
    PADDLE_ENFORCE(var_descs_.count(name) == 1, "%s must be in Block", name);
    return var_descs_.at(name);
  }

  bool HasVar(const std::string& name) const {
    return var_descs_.count(name) > 0;
  }

 private:
  const std::map<std::string, VarDesc*>& var_descs_;
};

class CompileTimeInferShapeContext : public InferShapeContextBase {
 public:
  CompileTimeInferShapeContext(const OperatorBase& op,
                               const BlockDesc& block_desc)
      : op_(op), block_desc_(block_desc) {}

  bool HasInput(const std::string& name) const {
    return block_desc_.HasVar(name);
  }

  bool HasOutput(const std::string& name) const {
    return block_desc_.HasVar(name);
  }

  DDim GetInputDim(const std::string& name) const {
    return GetDim(op_.Input(name));
  }

  void SetInputDim(const std::string& name, const DDim& dim) const {
    SetDim(op_.Input(name), dim);
  }

  DDim GetOutputDim(const std::string& name) const {
    return GetDim(op_.Output(name));
  }

  void SetOutputDim(const std::string& name, const DDim& dim) const {
    SetDim(op_.Output(name), dim);
  }

  AttrReader Attrs() const { return AttrReader(op_.Attrs()); }

  const std::vector<std::string>& Inputs(const std::string& name) const {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(const std::string& name) const {
    return op_.Outputs(name);
  }

 private:
  DDim GetDim(const std::string& name) const {
    VarDesc* desc = block_desc_.GetVar(name);
    std::vector<int64_t> dim;
    int length = desc->lod_tensor().dims().size();
    dim.reserve(length);
    std::copy(desc->lod_tensor().dims().begin(),
              desc->lod_tensor().dims().end(), std::back_inserter(dim));
    return make_ddim(dim);
  }

  void SetDim(const std::string& name, const DDim& dim) const {
    VarDesc* desc = block_desc_.GetVar(name);
    auto tensor = desc->mutable_lod_tensor();
    tensor->clear_dims();
    for (int i = 0; i < dim.size(); ++i) {
      tensor->add_dims(static_cast<int>(dim[i]));
    }
  }

  const OperatorBase& op_;
  const BlockDesc& block_desc_;
};

class RunTimeInferShapeContext : public InferShapeContextBase {
 public:
  RunTimeInferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  bool HasInput(const std::string& name) const {
    auto ipt = op_.Input(name);
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  bool HasOutput(const std::string& name) const {
    auto ipt = op_.Output(name);
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  DDim GetInputDim(const std::string& name) const {
    return GetDim(op_.Input(name));
  }

  void SetInputDim(const std::string& name, const DDim& dim) const {
    SetDim(op_.Input(name), dim);
  }

  DDim GetOutputDim(const std::string& name) const {
    return GetDim(op_.Output(name));
  }

  void SetOutputDim(const std::string& name, const DDim& dim) const {
    SetDim(op_.Output(name), dim);
  }

  AttrReader Attrs() const { return AttrReader(op_.Attrs()); }

  const std::vector<std::string>& Inputs(const std::string& name) const {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(const std::string& name) const {
    return op_.Outputs(name);
  }

 private:
  DDim GetDim(const std::string& name) const {
    return GetTensorFromVar(scope_.FindVar(name))->dims();
  }

  void SetDim(const std::string& name, const DDim& dim) const {
    Tensor* t;
    auto* var = scope_.FindVar(name);
    if (!var->IsType<LoDTensor>() && !var->IsType<Tensor>()) {
      t = var->GetMutable<LoDTensor>();
    } else {
      t = const_cast<Tensor*>(GetTensorFromVar(scope_.FindVar(name)));
    }
    t->Resize(dim);
  }

  const OperatorBase& op_;
  const Scope& scope_;
};

class OpKernel {
 public:
  /**
   * ExecutionContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * ExecutionContext. User should construct it before run the Operator.
   */

  virtual void Compute(const ExecutionContext& context) const = 0;

  virtual ~OpKernel() {}
};

class OperatorWithKernel : public OperatorBase {
 public:
  struct OpKernelKey {
    platform::Place place_;

    OpKernelKey() = default;
    explicit OpKernelKey(const platform::DeviceContext& dev_ctx) {
      place_ = dev_ctx.GetPlace();
    }

    bool operator==(const OpKernelKey& o) const {
      return platform::places_are_same_class(place_, o.place_);
    }
  };

  struct OpKernelHash {
    std::hash<bool> hash_;
    size_t operator()(const OpKernelKey& key) const {
      return hash_(platform::is_gpu_place(key.place_));
    }
  };

  using OpKernelMap =
      std::unordered_map<OpKernelKey, std::unique_ptr<OpKernel>, OpKernelHash>;

  OperatorWithKernel(const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  // runtime infershape
  void InferShape(const Scope& scope) const override {
    InferShape(RunTimeInferShapeContext(*this, scope));
  }

  // compile time infershape
  void InferShape(const BlockDesc& block_desc) const {
    InferShape(CompileTimeInferShapeContext(*this, block_desc));
  }

  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const final {
    auto& opKernel = AllOpKernels().at(type_).at(OpKernelKey(dev_ctx));
    opKernel->Compute(ExecutionContext(*this, scope, dev_ctx));
  }

  static std::unordered_map<std::string /* op_type */, OpKernelMap>&
  AllOpKernels() {
    static std::unordered_map<std::string, OpKernelMap> g_all_op_kernels;
    return g_all_op_kernels;
  }

  bool SupportGPU() const override {
    OperatorWithKernel::OpKernelKey key;
    key.place_ = platform::GPUPlace();
    return OperatorWithKernel::AllOpKernels().at(type_).count(key) != 0;
  }

 protected:
  virtual void InferShape(const InferShapeContext& ctx) const {}
  virtual void InferShape(const InferShapeContextBase& ctx) const {}
};

}  // namespace framework
}  // namespace paddle
