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

#include "paddle/framework/attribute.h"
#include "paddle/framework/op_desc.pb.h"
#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/scope.h"
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

/**
 * OperatorBase has the basic element that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class OperatorBase {
 public:
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
  virtual void InferShape(const Scope& scope) const = 0;

  /// Net will call this function to Run an op.
  virtual void Run(const Scope& scope,
                   const platform::DeviceContext& dev_ctx) const = 0;

  virtual bool IsNetOp() const { return false; }

  virtual bool SupportGPU() const { return false; }

  /// rename inputs outputs name
  void Rename(const std::string& old_name, const std::string& new_name);

  //! Get a input with argument's name described in `op_proto`
  const std::string& Input(const std::string& name) const;

  //! Get a input which has multiple variables.
  //! TODO add a vector_view to prevent memory copy.
  std::vector<std::string> Inputs(const std::string& name) const;
  //! Get a output with argument's name described in `op_proto`
  const std::string& Output(const std::string& name) const;
  //! Get an output which has multiple variables.
  //! TODO add a vector_view to prevent memory copy.
  std::vector<std::string> Outputs(const std::string& name) const;

 public:
  std::string type_;
  // NOTE: in case of OpGrad, inputs_ contains:
  // I (Inputs)
  // O (Outputs)
  // OG (Output Gradients)
  std::vector<std::string> inputs_;
  // NOTE: in case of OpGrad, outputs_ contains
  // IG (Inputs Gradients)
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
  // store the arguments' offset described in op_desc.
  std::shared_ptr<std::unordered_map<std::string, int>> in_out_idxs_;
};

class InferShapeContext {
 public:
  InferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  size_t InputSize() const { return op_.inputs_.size(); }

  size_t OutputSize() const { return op_.outputs_.size(); }

  const Variable* InputVar(const size_t index) const {
    return scope_.FindVar(op_.inputs_.at(index));
  }

  Variable* OutputVar(const size_t index) const {
    return scope_.FindVar(op_.outputs_.at(index));
  }

  const Variable* InputVar(const std::string& name) const {
    return scope_.FindVar(op_.Input(name));
  }

  Variable* OutputVar(const std::string& name) const {
    return scope_.FindVar(op_.Output(name));
  }

  const std::vector<const Variable*> MultiInputVar(
      const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const Variable*> res;
    res.reserve(names.size());
    std::transform(
        names.begin(), names.end(), std::back_inserter(res),
        [this](const std::string& name) { return scope_.FindVar(name); });
    return res;
  }

  std::vector<const Variable*> MultiOutputVar(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<const Variable*> res;
    res.reserve(names.size());
    std::transform(
        names.begin(), names.end(), std::back_inserter(res),
        [this](const std::string& name) { return scope_.FindVar(name); });
    return res;
  }

  template <typename T>
  const T* Input(const size_t index) const {
    auto var = InputVar(index);
    PADDLE_ENFORCE_NOT_NULL(var, "Input(%d) should not be nullptr", index);
    return &var->Get<T>();
  }

  template <typename T>
  T* Output(const size_t index) const {
    auto var = OutputVar(index);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        "Output(%d) not be nullptr, which means variable [%s] does not "
        "exist in scope",
        index, op_.outputs_[index]);
    return var->GetMutable<T>();
  }

  template <typename T>
  const T* Input(const std::string& name) const {
    auto var = InputVar(name);
    PADDLE_ENFORCE_NOT_NULL(var, "Input(%s) should not be nullptr", name);
    return &var->Get<T>();
  }

  template <typename T>
  T* Output(const std::string& name) const {
    auto var = OutputVar(name);
    PADDLE_ENFORCE_NOT_NULL(var, "Output(%s) should not be nullptr", name);
    return var->GetMutable<T>();
  }

  template <typename T>
  const std::vector<const T*> MultiInput(const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const T*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [&](const std::string& sub_name) {
                     auto var = scope_.FindVar(sub_name);
                     PADDLE_ENFORCE_NOT_NULL(
                         var, "MultiInput(%s:%s) should not be nullptr", name,
                         sub_name);
                     return &var->Get<T>();
                   });
    return res;
  }

  template <typename T>
  std::vector<const T*> MultiOutput(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<const T*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [&](const std::string& sub_name) {
                     auto var = scope_.FindVar(sub_name);
                     PADDLE_ENFORCE_NOT_NULL(
                         var, "MultiOutput(%s:%s) should not be nullptr", name,
                         sub_name);
                     return var->GetMutable<T>();
                   });
    return res;
  }

  const OperatorBase& op_;
  const Scope& scope_;
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

class ExecutionContext : public InferShapeContext {
 public:
  ExecutionContext(const OperatorBase& op, const Scope& scope,
                   const platform::DeviceContext* device_context)
      : InferShapeContext(op, scope), device_context_(device_context) {}

  template <typename PlaceType,
            typename DeviceType =
                typename EigenDeviceConverter<PlaceType>::EigenDeviceType>
  DeviceType& GetEigenDevice() const;

  platform::Place GetPlace() const { return device_context_->GetPlace(); }

  const platform::DeviceContext* device_context_;
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

  void InferShape(const Scope& scope) const override {
    InferShape(InferShapeContext(*this, scope));
  }

  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const final {
    auto& opKernel = AllOpKernels().at(type_).at(OpKernelKey(dev_ctx));
    opKernel->Compute(ExecutionContext(*this, scope, &dev_ctx));
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
  virtual void InferShape(const InferShapeContext& ctx) const = 0;
};

}  // namespace framework
}  // namespace paddle
