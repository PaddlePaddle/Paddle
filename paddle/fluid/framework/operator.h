/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"  // For VLOG
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/variant.h"

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

// define some kernel priority
/* Define multiple kernel type fallback order*/
extern std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority;

inline std::string GradVarName(const std::string& var_name) {
  return var_name + kGradVarSuffix;
}

proto::VarType::Type GetDataTypeOfVar(const Variable* var);

class OperatorBase;
class ExecutionContext;

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

  /// Executor will call this interface function to Run an op.
  //  The implementation should be written at RunImpl
  void Run(const Scope& scope, const platform::Place& place);

  // FIXME(typhoonzero): this is only used for recv_op to stop event_loop.
  virtual void Stop() {}

  /// if scope is not null, also show dimensions of arguments
  virtual std::string DebugStringEx(const Scope* scope) const;
  std::string DebugString() const { return DebugStringEx(nullptr); }

  virtual bool SupportGPU() const { return false; }

  const std::string& Type() const { return type_; }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }
  const AttributeMap& Attrs() const { return attrs_; }

  const VariableNameMap& Inputs() const { return inputs_; }
  const VariableNameMap& Outputs() const { return outputs_; }

  bool HasInputs(const std::string& name) const;
  //! Get a input with argument's name described in `op_proto`
  std::string Input(const std::string& name) const;
  //! Get a input which has multiple variables.
  const std::vector<std::string>& Inputs(const std::string& name) const;
  //! Get all inputs variable names
  std::vector<std::string> InputVars() const;

  bool HasOutputs(const std::string& name) const;
  //! Get a output with argument's name described in `op_proto`
  std::string Output(const std::string& name) const;
  //! Get an output which has multiple variables.
  //! TODO add a vector_view to prevent memory copy.
  const std::vector<std::string>& Outputs(const std::string& name) const;
  //! Get all outputs variable names
  virtual std::vector<std::string> OutputVars(bool has_intermediate) const;

  // Return a new operator instance, which is as same as this.
  // Use unique_ptr to prevent caller forget to delete this pointer.
  virtual std::unique_ptr<OperatorBase> Clone() const = 0;

 protected:
  std::string type_;
  // NOTE: in case of OpGrad, inputs_ contains:
  // I (Inputs)
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
  virtual void RunImpl(const Scope& scope,
                       const platform::Place& place) const = 0;
};

// Macro for define a clone method.
// If you are writing an kernel operator, `Clone` will be defined when you
// register it. i.e. `Clone` method is not needed to define by yourself.
#define DEFINE_OP_CLONE_METHOD(cls)                                            \
  std::unique_ptr<::paddle::framework::OperatorBase> Clone() const final {     \
    return std::unique_ptr<::paddle::framework::OperatorBase>(new cls(*this)); \
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
  std::unique_ptr<OperatorBase> Clone() const override {
    return std::unique_ptr<OperatorBase>(new NOP(*this));
  }

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {}
};

class ExecutionContext {
 public:
  ExecutionContext(const OperatorBase& op, const Scope& scope,
                   const platform::DeviceContext& device_context)
      : op_(op), scope_(scope), device_context_(device_context) {}

  const OperatorBase& op() const { return op_; }

  const Scope& scope() const { return scope_; }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return op_.Attr<T>(name);
  }

  bool HasInput(const std::string& name) const;

  bool HasOutput(const std::string& name) const;

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

  std::vector<Variable*> MultiOutputVar(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<Variable*> res;
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

  platform::Place GetPlace() const { return device_context_.GetPlace(); }

  template <typename DeviceContextType>
  const DeviceContextType& device_context() const {
    return *reinterpret_cast<const DeviceContextType*>(&device_context_);
  }

  const platform::DeviceContext& device_context() const {
    return device_context_;
  }

#ifdef PADDLE_WITH_CUDA
  const inline platform::CUDADeviceContext& cuda_device_context() const {
    PADDLE_ENFORCE(platform::is_gpu_place(device_context_.GetPlace()));
    return *reinterpret_cast<const platform::CUDADeviceContext*>(
        &device_context_);
  }
#endif

  //! Get actual name vector for this input.
  const std::vector<std::string>& Inputs(const std::string& name) const {
    return op_.Inputs(name);
  }

  //! Get actual name vector for this output.
  const std::vector<std::string>& Outputs(const std::string& name) const {
    return op_.Outputs(name);
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
  const platform::DeviceContext& device_context_;
};

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const;

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const;

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const;

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const;

class OpKernelBase {
 public:
  /**
   * ExecutionContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * ExecutionContext. User should construct it before run the Operator.
   */

  virtual void Compute(const ExecutionContext& context) const = 0;

  virtual ~OpKernelBase() = default;
};

template <typename T>
class OpKernel : public OpKernelBase {
 public:
  using ELEMENT_TYPE = T;
};

class OperatorWithKernel : public OperatorBase {
 public:
  using OpKernelMap =
      std::unordered_map<OpKernelType, std::unique_ptr<OpKernelBase>,
                         OpKernelType::Hash>;

  OperatorWithKernel(const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  static std::unordered_map<std::string /* op_type */, OpKernelMap>&
  AllOpKernels() {
    static std::unordered_map<std::string, OpKernelMap> g_all_op_kernels;
    return g_all_op_kernels;
  }

  bool SupportGPU() const override {
    auto& op_kernels = OperatorWithKernel::AllOpKernels().at(type_);
    return std::any_of(op_kernels.begin(), op_kernels.end(),
                       [](OpKernelMap::const_reference kern_pair) {
                         return platform::is_gpu_place(kern_pair.first.place_);
                       });
  }

  virtual void InferShape(InferShapeContext* ctx) const {
    OpInfoMap::Instance().Get(Type()).infer_shape_(ctx);
  }

 protected:
  virtual OpKernelType GetExpectedKernelType(const ExecutionContext& ctx) const;
  virtual OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const OpKernelType& expected_kernel_type) const;

 private:
  // indicate kernel DataType by input data. By default all input data must be
  // same.
  proto::VarType::Type IndicateDataType(const ExecutionContext& ctx) const;
  void RunImpl(const Scope& scope, const platform::Place& place) const final;
};

extern bool OpSupportGPU(const std::string& op_type);

}  // namespace framework
}  // namespace paddle
