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
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"  // For VLOG
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator_kernel_configs.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/variant.h"

DECLARE_int32(inner_op_parallelism);

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

constexpr size_t kGradVarSuffixSize = 5U;

/// Variables with this suffix are supposed to be filled up with zeros.
constexpr char kZeroVarSuffix[] = "@ZERO";

/// Variables with this suffix are the new Gradient.
constexpr char kNewGradSuffix[] = "@NEWGRAD@";

/// RuntimeContext is used to relate input/output names of Operator with
/// the corresponding variables in name scope.
/// If an Op has attribute kEnableCacheRuntimeContext, it means that in a same
/// name scope, since the input/output names of this Op do not change in the
/// execution, RuntimeContext could be created only at the first iteration of
/// this Op's execution to save the elapsed time.
constexpr char kEnableCacheRuntimeContext[] = "@ENABLE_CACHE_RUNTIME_CONTEXT@";

/// If an Op has this attribute, all its kernels should calculate output
/// variable's shape in the corresponding Compute() function. And
/// OperatorWithKernel::RunImpl() would skip call this Op's InferShape()
/// function in its runtime for speedup.
/// TODO(luotao): Note that this temporal attribute would be deleted after all
/// ops contain it.
constexpr char kAllKernelsMustComputeRuntimeShape[] =
    "@ALL_KERNELS_MUST_COMPUTE_RUNTIME_SHAPE@";

// define some kernel priority
/* Define multiple kernel type fallback order*/
extern std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority;

inline std::string GradVarName(const std::string& var_name) {
  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

inline std::string GradOriginalVarName(const std::string& grad_var_name) {
  std::size_t pos = grad_var_name.rfind(kGradVarSuffix);
  if (pos == std::string::npos) {
    return grad_var_name;
  } else {
    return grad_var_name.substr(0, pos);
  }
}

proto::VarType::Type GetDataTypeOfVar(const Variable* var);
const Tensor* GetLoDTensorOrSelectedRowsValueFromVar(const Variable& var);
Tensor* GetMutableLoDTensorOrSelectedRowsValueFromVar(Variable* var);

class OperatorBase;
class ExecutionContext;

class RuntimeContext {
 public:
  RuntimeContext(const VariableNameMap& innames,
                 const VariableNameMap& outnames, const Scope& scope);

  RuntimeContext(const VariableValueMap& invars,
                 const VariableValueMap& outvars)
      : inputs(invars), outputs(outvars) {}

  VariableValueMap inputs;
  VariableValueMap outputs;
};

/**
 * OperatorBase has the basic elements that Net will call to do computation.
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

  bool HasAttr(const std::string& name) const { return attrs_.count(name); }
  template <typename T>
  inline const T& Attr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.find(name) != attrs_.end(),
                   "%s should be in AttributeMap", name);
    return boost::get<T>(attrs_.at(name));
  }
  const AttributeMap& Attrs() const { return attrs_; }

  const VariableNameMap& Inputs() const { return inputs_; }
  const VariableNameMap& Outputs() const { return outputs_; }

  const OpInfo& Info() const {
    PADDLE_ENFORCE_NOT_NULL(info_, "OpInfo of %s is not found", type_);
    return *info_;
  }

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

  void SetIsCalledByExecutor(bool x) { run_by_executor_ = x; }
  virtual void RuntimeInferShape(const Scope& scope,
                                 const platform::Place& place,
                                 const RuntimeContext& ctx) const {}

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

  // OpInfo
  const OpInfo* info_;

  // Whether this operator executes in an Executor.
  bool run_by_executor_{true};

 private:
  void GenerateTemporaryNames();
  void CheckAllInputOutputSet() const;
  virtual void RunImpl(const Scope& scope,
                       const platform::Place& place) const = 0;
};

#ifdef PADDLE_WITH_CUDA
using KernelConfig = boost::variant<
    std::shared_ptr<AlgorithmsCache<cudnnConvolutionFwdAlgo_t>>,
    std::shared_ptr<AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>>,
    std::shared_ptr<AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>>>;
#else
using KernelConfig = boost::variant<boost::blank>;
#endif

using OpKernelConfigsMap =
    std::unordered_map<OpKernelType, std::vector<KernelConfig>,
                       OpKernelType::Hash>;

class ExecutionContext {
 public:
  ExecutionContext(const OperatorBase& op, const Scope& scope,
                   const platform::DeviceContext& device_context,
                   const RuntimeContext& ctx,
                   std::vector<KernelConfig>* configs)
      : op_(op),
        scope_(scope),
        device_context_(device_context),
        ctx_(ctx),
        kernel_configs_(configs) {}

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

  const Variable* InputVar(const std::string& name) const;

  Variable* OutputVar(const std::string& name) const;

  const std::vector<const Variable*> MultiInputVar(
      const std::string& name) const {
    auto it = ctx_.inputs.find(name);
    if (it == ctx_.inputs.end()) {
      return {};
    }
    return {it->second.begin(), it->second.end()};
  }

  std::vector<Variable*> MultiOutputVar(const std::string& name) const {
    auto names = op_.Outputs(name);
    auto it = ctx_.outputs.find(name);
    if (it == ctx_.outputs.end()) {
      return {};
    }
    return it->second;
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
    auto it = ctx_.inputs.find(name);
    if (it == ctx_.inputs.end()) {
      return {};
    }
    const std::vector<Variable*>& vars = it->second;
    std::vector<const T*> res;
    res.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                   [&](Variable* var) -> const T* {
                     return var == nullptr ? nullptr : &var->Get<T>();
                   });
    return res;
  }

  template <typename T>
  std::vector<T*> MultiOutput(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    if (it == ctx_.outputs.end()) {
      return {};
    }
    const std::vector<Variable*>& vars = it->second;
    std::vector<T*> res;
    res.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                   [&](Variable* var) -> T* {
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

  template <typename T, typename DevContext>
  Tensor AllocateTmpTensor(const framework::DDim& dim,
                           const DevContext& dev_ctx) const {
    auto tmp_allocation_ptr = platform::DeviceTemporaryAllocator::Instance()
                                  .Get<DevContext>(dev_ctx)
                                  .Allocate(product(dim) * sizeof(T));
    auto& deleter = tmp_allocation_ptr.get_deleter();
    auto* allocation_ptr = tmp_allocation_ptr.release();
    auto shared_allocation = std::shared_ptr<memory::allocation::Allocation>(
        allocation_ptr, deleter);

    PADDLE_ENFORCE_GE(allocation_ptr->size(),
                      framework::product(dim) * sizeof(T));

    paddle::framework::Tensor temp_tensor(
        framework::ToDataType(std::type_index(typeid(T))));
    temp_tensor.Resize(dim);
    temp_tensor.ResetHolder(std::move(shared_allocation));
    return temp_tensor;
  }

  template <typename T>
  T& GetKernelConfig(size_t idx) const {
    PADDLE_ENFORCE(
        kernel_configs_ && kernel_configs_->size() > static_cast<size_t>(idx),
        "%s selected kernel doesn't have kernel config %lu <= %lu",
        op_.Type().c_str(), kernel_configs_->size(), idx);
    return *boost::get<std::shared_ptr<T>>((*kernel_configs_)[idx]);
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
  const platform::DeviceContext& device_context_;
  const RuntimeContext& ctx_;
  mutable std::vector<KernelConfig>* kernel_configs_;
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
  using OpKernelFunc = std::function<void(const ExecutionContext&)>;
  using OpKernelMap =
      std::unordered_map<OpKernelType, OpKernelFunc, OpKernelType::Hash>;

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
    Info().infer_shape_(ctx);
  }

  void RuntimeInferShape(const Scope& scope, const platform::Place& place,
                         const RuntimeContext& ctx) const override;

  virtual OpKernelType GetExpectedKernelType(const ExecutionContext& ctx) const;

  std::vector<KernelConfig>* GetKernelConfig(const OpKernelType& key) const;

 protected:
  virtual OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const OpKernelType& expected_kernel_type) const;

 private:
  // indicate kernel DataType by input data. By default all input data must be
  // same.
  proto::VarType::Type IndicateDataType(const ExecutionContext& ctx) const;
  void RunImpl(const Scope& scope, const platform::Place& place) const final;
  void RunImpl(const Scope& scope, const platform::Place& place,
               RuntimeContext* runtime_ctx) const;

  /**
   * Transfer data from scope to a transfered scope. If there is no data need to
   * be tranfered, it returns nullptr.
   *
   * * transfered_inplace_vars is a output vector.
   */
  Scope* PrepareData(const Scope& scope,
                     const OpKernelType& expected_kernel_key,
                     std::vector<std::string>* transfered_inplace_vars,
                     RuntimeContext* ctx) const;

  void TransferInplaceVarsBack(const Scope& scope,
                               const std::vector<std::string>& inplace_vars,
                               const Scope& exec_scope) const;

  void ChooseKernel(const RuntimeContext& ctx, const Scope& scope,
                    const platform::Place& place) const;

 protected:
  mutable OpKernelConfigsMap kernel_configs_map_;
  mutable std::unique_ptr<OpKernelType> kernel_type_;
  mutable std::unique_ptr<OpKernelFunc> kernel_func_;
  mutable std::unique_ptr<RuntimeContext> runtime_ctx_;
  mutable const Scope* pre_scope_ = nullptr;
  mutable bool enable_cache_runtime_context = false;
  mutable bool all_kernels_must_compute_runtime_shape = false;
  mutable std::mutex cache_update_mutex_;
};

extern bool OpSupportGPU(const std::string& op_type);

}  // namespace framework
}  // namespace paddle
