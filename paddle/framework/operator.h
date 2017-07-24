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

#include <boost/variant.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/framework/attr_checker.h"
#include "paddle/framework/op_desc.pb.h"
#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/place.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace framework {

class OperatorBase;
class InferShapeContext;
class KernelContext;
/**
 * OperatorBase has the basic element that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class OperatorBase {
 public:
  /// If a variable is a empty variable, that name will be used.
  static std::string EMPTY_VAR_NAME() { return "@EMPTY@"; }

  /// If a variable is a temporary variable, that name will be set in Python,
  /// but it will be convert to a unique name in scope after OpCreator.
  static std::string TMP_VAR_NAME() { return "@TEMP@"; }

  /// If a variable's name has a certain suffix, it means that the
  /// variable is the gradient of another varibale.
  /// e.g. Variable "x@GRAD" is the gradient of varibale "x".
  static std::string GRAD_VAR_SUFFIX() { return "@GRAD"; }

  virtual ~OperatorBase() {}

  template <typename T>
  inline const T& GetAttr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }

  virtual std::string DebugString() const;

  /// Init will be called after CreateOperator, you can put some initialization
  /// logic here.
  virtual void Init() {}

  /// InferShape infer the size of Variables used by this Operator with
  /// information inside scope
  virtual void InferShape(const std::shared_ptr<Scope>& scope) const = 0;
  virtual void InferShapeImpl(const InferShapeContext& ctx) const = 0;

  /// Net will call this function to Run an op.
  virtual void Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const = 0;

  // Get a input with argument's name described in `op_proto`
  const std::string& Input(const std::string& name) const;
  // Get a input which has multiple variables.
  // TODO add a vector_view to prevent memory copy.
  std::vector<std::string> Inputs(const std::string& name) const;
  // Get a output with argument's name described in `op_proto`
  const std::string& Output(const std::string& name) const;
  // Get an output which has multiple variables.
  // TODO add a vector_view to prevent memory copy.
  std::vector<std::string> Outputs(const std::string& name) const;

 public:
  std::string type_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
  // store the arguments' offset described in op_desc.
  std::shared_ptr<std::unordered_map<std::string, int>> in_out_idxs_;
};

class OperatorContext {
 public:
  OperatorContext(const OperatorBase* op, const std::shared_ptr<Scope>& scope)
      : op_(*op), scope_(scope) {}

  int InputSize() const { return static_cast<int>(op_.inputs_.size()); }

  int OutputSize() const { return static_cast<int>(op_.outputs_.size()); }

  const Variable* InputVar(int index) const {
    return scope_->GetVariable(op_.inputs_[index]);
  }

  Variable* OutputVar(int index) const {
    return scope_->GetVariable(op_.outputs_[index]);
  }

  const Variable* InputVar(const std::string& name) const {
    return scope_->GetVariable(op_.Input(name));
  }

  Variable* OutputVar(const std::string& name) const {
    return scope_->GetVariable(op_.Output(name));
  }

  const std::vector<const Variable*> InputVars(const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const Variable*> res;
    std::transform(
        names.begin(), names.end(), res.begin(),
        [this](const std::string& name) { return scope_->GetVariable(name); });
    return res;
  }

  const std::vector<const Variable*> OutputVars(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<const Variable*> res;
    std::transform(
        names.begin(), names.end(), res.begin(),
        [this](const std::string& name) { return scope_->GetVariable(name); });
    return res;
  }

  const Tensor& Input(int index) const {
    return InputVar(index)->Get<Tensor>();
  }

  Tensor* Output(int index) const {
    return OutputVar(index)->GetMutable<Tensor>();
  }

  const Tensor& Input(const std::string& name) const {
    return InputVar(name)->Get<Tensor>();
  }

  Tensor* Output(const std::string& name) const {
    return OutputVar(name)->GetMutable<Tensor>();
  }

  const std::vector<const Tensor*> Inputs(const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const Tensor*> res;
    std::transform(names.begin(), names.end(), res.begin(),
                   [this](const std::string& name) {
                     return &scope_->GetVariable(name)->Get<Tensor>();
                   });
    return res;
  }

  std::vector<const Tensor*> Outputs(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<const Tensor*> res;
    std::transform(names.begin(), names.end(), res.begin(),
                   [this](const std::string& name) {
                     return scope_->GetVariable(name)->GetMutable<Tensor>();
                   });
    return res;
  }

  const OperatorBase& op_;
  const std::shared_ptr<Scope>& scope_;
};

class InferShapeContext : public OperatorContext {
 public:
  InferShapeContext(const OperatorBase* op, const std::shared_ptr<Scope>& scope)
      : OperatorContext(op, scope) {}
};

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

class KernelContext : public OperatorContext {
 public:
  KernelContext(const OperatorBase* op, const std::shared_ptr<Scope>& scope,
                const platform::DeviceContext& device_context)
      : OperatorContext(op, scope), device_context_(device_context) {}

  template <typename PlaceType,
            typename DeviceType =
                typename EigenDeviceConverter<PlaceType>::EigenDeviceType>
  DeviceType* GetEigenDevice() const;

  platform::Place GetPlace() const { return device_context_.GetPlace(); }

  const platform::DeviceContext& device_context_;
};

class OpKernel {
 public:
  /**
   * KernelContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * KernelContext. User should construct it before run the Operator.
   */

  virtual void Compute(const KernelContext& context) const = 0;

  virtual ~OpKernel() {}
};

class OperatorWithKernel : public OperatorBase {
 public:
  struct OpKernelKey {
    platform::Place place_;

    OpKernelKey() = default;
    OpKernelKey(const platform::DeviceContext& dev_ctx) {
      place_ = dev_ctx.GetPlace();
    }

    bool operator==(const OpKernelKey& o) const { return place_ == o.place_; }
  };

  struct OpKernelHash {
    std::hash<bool> hash_;
    size_t operator()(const OpKernelKey& key) const {
      return hash_(platform::is_gpu_place(key.place_));
    }
  };

  using OpKernelMap =
      std::unordered_map<OpKernelKey, std::unique_ptr<OpKernel>, OpKernelHash>;

  void Run(const std::shared_ptr<Scope>& scope,
           const platform::DeviceContext& dev_ctx) const final {
    auto& opKernel = AllOpKernels().at(type_).at(OpKernelKey(dev_ctx));
    opKernel->Compute(KernelContext(this, scope, dev_ctx));
  }

  static std::unordered_map<std::string /* op_type */, OpKernelMap>&
  AllOpKernels() {
    static std::unordered_map<std::string, OpKernelMap> g_all_op_kernels;
    return g_all_op_kernels;
  }
};

}  // namespace framework
}  // namespace paddle
