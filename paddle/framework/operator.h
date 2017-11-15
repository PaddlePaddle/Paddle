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
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"  // For VLOG
#include "paddle/framework/attribute.h"
#include "paddle/framework/block_desc.h"
#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_info.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/selected_rows.h"
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

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }

  virtual std::string DebugString() const;

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
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {}
  std::unique_ptr<OperatorBase> Clone() const override {
    return std::unique_ptr<OperatorBase>(new NOP(*this));
  }
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

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const {
    PADDLE_ENFORCE_LT(i, InputSize(in));
    PADDLE_ENFORCE_LT(j, OutputSize(out));
    auto* in_var = MultiInputVar(in)[i];
    auto* out_var = MultiOutputVar(out)[j];
    if (!in_var->IsType<LoDTensor>()) return;
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());
  }

  template <typename PlaceType,
            typename DeviceType = typename platform::EigenDeviceConverter<
                PlaceType>::EigenDeviceType>
  DeviceType& GetEigenDevice() const;

  platform::Place GetPlace() const { return device_context_.GetPlace(); }

  const platform::DeviceContext& device_context() const {
    return device_context_;
  }

  //! Get actual name vector for this input.
  const std::vector<std::string>& Inputs(const std::string& name) const {
    return op_.Inputs(name);
  }

  //! Get actual name vector for this output.
  const std::vector<std::string>& Outputs(const std::string& name) const {
    return op_.Outputs(name);
  }

#ifdef PADDLE_WITH_CUDA
  const inline platform::CUDADeviceContext& cuda_device_context() const {
    PADDLE_ENFORCE(platform::is_gpu_place(device_context_.GetPlace()));
    return *reinterpret_cast<const platform::CUDADeviceContext*>(
        &device_context_);
  }
#endif

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

struct OpKernelType {
  struct Hash {
    std::hash<int> hash_;
    size_t operator()(const OpKernelType& key) const {
      int place = key.place_.which();
      int data_type = static_cast<int>(key.data_type_);
      int pre_hash = data_type << NUM_PLACE_TYPE_LIMIT_IN_BIT |
                     (place & ((1 << NUM_PLACE_TYPE_LIMIT_IN_BIT) - 1));
      return hash_(pre_hash);
    }
  };

  platform::Place place_;
  DataType data_type_;

  OpKernelType(DataType data_type, platform::Place place)
      : place_(place), data_type_(data_type) {}

  OpKernelType(DataType data_type, const platform::DeviceContext& dev_ctx)
      : place_(dev_ctx.GetPlace()), data_type_(data_type) {}

  bool operator==(const OpKernelType& o) const {
    return platform::places_are_same_class(place_, o.place_) &&
           data_type_ == o.data_type_;
  }
};

class OperatorWithKernel : public OperatorBase {
 public:
  using OpKernelMap =
      std::unordered_map<OpKernelType, std::unique_ptr<OpKernelBase>,
                         OpKernelType::Hash>;

  OperatorWithKernel(const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const final;

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
  virtual OpKernelType GetKernelType(const ExecutionContext& ctx) const;

 private:
  // indicate kernel DataType by input data. Defaultly all input data must be
  // same.
  DataType IndicateDataType(const ExecutionContext& ctx) const;
};

std::ostream& operator<<(std::ostream& os, const OpKernelType& kernel_key);

extern bool OpSupportGPU(const std::string& op_type);

}  // namespace framework
}  // namespace paddle
