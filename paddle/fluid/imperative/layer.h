// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <map>     // NOLINT
#include <memory>  // NOLINT
#include <mutex>   // NOLINT
#include <set>
#include <string>         // NOLINT
#include <unordered_map>  // NOLINT
#include <utility>
#include <vector>  // NOLINT

// clang-format off
#include "paddle/fluid/framework/python_headers.h"
// clang-format on

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/flags.h"

namespace paddle {
namespace imperative {

class VarBase;

namespace py = ::pybind11;

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             framework::OperatorWithKernel::OpKernelFunc func,
             platform::DeviceContext* dev_ctx,
             std::vector<framework::KernelConfig>* kernel_configs)
      : op(op),
        ctx(ctx),
        func(func),
        dev_ctx(dev_ctx),
        kernel_configs(kernel_configs) {}

  static PreparedOp Prepare(const framework::RuntimeContext& ctx,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);

    // check if op[type] has kernel registered.
    auto& all_op_kernels = op.AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op.Type());
    if (kernels_iter == all_op_kernels.end()) {
      PADDLE_THROW(
          "There are no kernels which are registered in the %s operator.",
          op.Type());
    }

    framework::OperatorWithKernel::OpKernelMap& kernels = kernels_iter->second;

    auto expected_kernel_key =
        op.GetExpectedKernelType(framework::ExecutionContext(
            op, framework::Scope(), *dev_ctx, ctx, nullptr));
    VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

    auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
    // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
    if (kernel_iter == kernels.end() &&
        expected_kernel_key.library_type_ == framework::LibraryType::kMKLDNN) {
      VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
      expected_kernel_key.library_type_ = framework::LibraryType::kPlain;
      expected_kernel_key.data_layout_ = framework::DataLayout::kAnyLayout;
      kernel_iter = kernels.find(expected_kernel_key);
    }
#endif
    if (kernel_iter == kernels.end()) {
      PADDLE_THROW("op %s does not have kernel for %s", op.Type(),
                   KernelTypeToString(expected_kernel_key));
    }
    std::vector<framework::KernelConfig>* kernel_configs =
        op.GetKernelConfig(expected_kernel_key);
    return PreparedOp(op, ctx, kernel_iter->second, dev_ctx, kernel_configs);
  }

  inline platform::DeviceContext* GetDeviceContext() const { return dev_ctx; }

  const framework::OperatorBase& op;
  const framework::RuntimeContext& ctx;
  framework::OperatorWithKernel::OpKernelFunc func;
  platform::DeviceContext* dev_ctx;
  std::vector<framework::KernelConfig>* kernel_configs;
};

class OpBase;

class ThreadSafeNameSet {
 public:
  void Insert(const std::string& name);

  void Remove(const std::string& name);

  std::vector<std::string> Names() const;

 private:
  std::multiset<std::string> set_;
  mutable std::mutex mtx_;
};

/* The wrapper for Variable which holds a Variable and a VarBase of its
 * gradient. This object should be managed totally by Python intepreter.
 *
 * Nearly all interface should be implemented in C++.
 */
class VarBase {
 public:
  static std::vector<std::string> AliveVarNames();

  // Internal interface, create VarBase from exist variable
  VarBase(const std::string& name, std::unique_ptr<framework::Variable> var,
          VarBase* grad, bool stop_gradient)
      : VarBase(name, var->Get<framework::LoDTensor>().type(),
                var->Get<framework::LoDTensor>().dims(),
                var->Get<framework::LoDTensor>().place(), nullptr, grad,
                stop_gradient, false, true) {
    var_ = std::move(var);
  }

  // Python interface
  VarBase(const std::string& name, const framework::proto::VarType::Type dtype,
          const std::vector<int64_t>& shape, const platform::Place& place,
          bool stop_gradient, bool persistable)
      : VarBase(name, dtype, framework::make_ddim(shape), place, stop_gradient,
                persistable) {}

  // Internal interface, create VarBase from with ddim
  VarBase(const std::string& name, const framework::proto::VarType::Type dtype,
          const framework::DDim& shape, const platform::Place& place,
          bool stop_gradient, bool persistable)
      : VarBase(name, dtype, shape, place, nullptr, nullptr, stop_gradient,
                persistable, true) {}

  // Grad used constructor
  VarBase(const std::string& name, const framework::proto::VarType::Type dtype,
          const std::vector<int64_t>& shape, const platform::Place& place,
          bool stop_gradient, bool persistable, bool need_initialize)
      : VarBase(name, dtype, framework::make_ddim(shape), place, nullptr,
                nullptr, stop_gradient, persistable, need_initialize) {}

 private:
  // TODO(minqiyang): need support SelectedRows
  VarBase(const std::string& name, framework::proto::VarType::Type dtype,
          const framework::DDim& shape, const platform::Place& place,
          std::unique_ptr<framework::Variable> var, VarBase* grad,
          bool stop_gradient, bool persistable, bool need_initialize)
      : name_(name),
        type_(framework::proto::VarType::LOD_TENSOR),
        place_(place),
        var_(std::move(var)),
        grads_(grad),
        dtype_(dtype),
        stop_gradient_(stop_gradient),
        persistable_(persistable),
        pre_op_(nullptr),
        pre_op_out_name_(),
        pre_op_out_idx_(-1) {
    if (!var_) {
      var_.reset(new framework::Variable());
    }

    auto tensor = var_->GetMutable<framework::LoDTensor>();
    tensor->Resize(shape);
    if (need_initialize) {
      tensor->mutable_data(place, dtype);
      is_initialized_ = true;
      VLOG(8) << "initialized varbase: " << name_ << " type: " << dtype
              << " place: " << place;
    } else {
      is_initialized_ = false;
      VLOG(8) << "not initialized varbase: " << name_;
    }
    VLOG(8) << "create varbase: " << name_ << " type: " << dtype
            << " place: " << place << "Stop gradient: " << stop_gradient_;

    if (IsDebugEnabled()) {
      name_set_.Insert(name_);
    }
  }

 public:
  virtual ~VarBase() {
    pre_op_ = nullptr;
    pre_op_out_idx_ = -1;
    VLOG(8) << "destruct varbase: " << name_;
    if (IsDebugEnabled()) {
      name_set_.Remove(name_);
    }
  }

  inline void SetName(const std::string& name) { name_ = name; }
  inline std::string Name() const { return name_; }
  inline bool IsInitialize() const { return is_initialized_; }
  inline void SetInitialize(bool inited) { is_initialized_ = inited; }
  inline std::vector<int64_t> Shape() const {
    if (var_->IsInitialized()) {
      return framework::vectorize(var_->Get<framework::LoDTensor>().dims());
    } else {
      return {};
    }
  }

  inline framework::DDim Dims() const {
    return var_->Get<framework::LoDTensor>().dims();
  }

  // data type. e.g.. FP32
  inline void SetDataType(framework::proto::VarType::Type type) {
    auto tensor = var_->GetMutable<framework::LoDTensor>();
    tensor->mutable_data(tensor->place(), type);
  }
  inline framework::proto::VarType::Type DataType() const { return dtype_; }

  // tensor type. e.g.. LoDTensor
  inline void SetType(framework::proto::VarType::Type type) { type_ = type; }
  inline framework::proto::VarType::Type Type() const { return type_; }

  inline void SetStopGradient(bool stop_gradient) {
    stop_gradient_ = stop_gradient;
    if (grads_) {
      grads_->stop_gradient_ = stop_gradient;
    }
  }
  inline bool IsStopGradient() const { return stop_gradient_; }

  inline void SetPersistable(bool persistable) { persistable_ = persistable; }
  inline bool IsPersistable() const { return persistable_; }
  inline void SetPreOp(OpBase* op) { pre_op_ = op; }
  inline platform::Place GetPlace() { return place_; }
  inline OpBase* PreOp() const { return pre_op_; }
  inline int PreOpOutIdx() const { return pre_op_out_idx_; }

  void RunBackward(const detail::BackwardStrategy& bck_stratedy);

  inline void ResetPreOp(OpBase* op) {
    if (op == pre_op_) {
      // clear pre_op info when op equals to var's pre_op
      pre_op_ = nullptr;
      pre_op_out_idx_ = -1;
    }
  }

  void InitBuffer() {
    if (!is_initialized_) {
      var_->GetMutable<framework::LoDTensor>()->mutable_data(place_, dtype_);
      is_initialized_ = true;
      VLOG(8) << "initialized varbase: " << name_ << " type: " << dtype_
              << " place: " << place_;
    } else {
      VLOG(8) << "var: " << name_ << " has already been initialized ";
    }
  }

  void TrackPreOp(OpBase* pre_op, const std::string& pre_op_out_name,
                  int pre_op_out_idx, bool pre_op_stop_gradient) {
    pre_op_ = pre_op;
    pre_op_out_name_ = pre_op_out_name;
    pre_op_out_idx_ = pre_op_out_idx;
    if (pre_op_stop_gradient) {
      stop_gradient_ = pre_op_stop_gradient;
    }
  }

  void ClearGradient() {
    VLOG(1) << "clear gradient of " << Name();
    if (grads_ && grads_->var_ && grads_->var_->IsInitialized()) {
      auto grads_t = grads_->var_->GetMutable<framework::LoDTensor>();
      operators::math::set_constant(
          *(platform::DeviceContextPool::Instance().Get(
              grads_->var_->Get<framework::LoDTensor>().place())),
          grads_t, 0.0);
    }
  }

  framework::LoDTensor& GradValue();

  std::unique_ptr<VarBase> NewVarBase(const platform::Place& dst_place,
                                      const bool blocking) const;

  inline std::string GradName() const {
    return string::Sprintf("%s@IGrad", Name());
  }

  std::string name_;
  framework::proto::VarType::Type type_;
  platform::Place place_;

  std::unique_ptr<framework::Variable> var_;
  std::shared_ptr<VarBase> grads_;

 private:
  framework::proto::VarType::Type dtype_;
  bool stop_gradient_;
  bool persistable_;
  bool is_initialized_;
  OpBase* pre_op_;
  std::string pre_op_out_name_;
  int pre_op_out_idx_;

  // A private flag to check memory leak
  static ThreadSafeNameSet name_set_;
};

/* The wrapper for OpDesc which holds a OpDesc and a OpDesc of its
 * gradient. This object should be managed totally by Python intepreter.
 */
class PYBIND11_HIDDEN OpBase {
 public:
  OpBase(const std::string& type)
      : type_(type),
        trace_id_(-1),
        place_(platform::CPUPlace()),
        backward_hooks_() {}

  virtual ~OpBase() {
    for (const auto& it : outputs_ref) {
      auto vb = it.lock();
      if (vb) {
        VLOG(3) << "Op reset by" << vb->name_;
        vb->ResetPreOp(this);
      }
    }
    // TODO(minqiyang): remove op_desc from block_desc in tracer
    // release resource
    for (framework::OpDesc* desc : grad_op_descs_) {
      delete desc;
    }
  }

  std::vector<VarBasePtrMap> ApplyGrad(
      BackwardSumMap* bck_map, GradientRef* grad_ref,
      const detail::BackwardStrategy& bck_stratedy);

  inline std::string Type() const { return type_; }
  inline std::string GradOpType(size_t index) const {
    PADDLE_ENFORCE_NOT_NULL(grad_op_descs_[index]);
    return grad_op_descs_[index]->Type();
  }

  void RegisterBackwardHooks(const py::object& callable);

  void InvokeBackwardHooks();

  void TrackPreOp(
      const std::string& inp_name,
      const std::vector<std::shared_ptr<imperative::VarBase>>& inputs) {
    auto& pre_ops_list = pre_ops_[inp_name];
    pre_ops_list.reserve(inputs.size());
    auto& pre_ops_out_idx_list = pre_ops_out_idx_[inp_name];
    for (std::shared_ptr<imperative::VarBase> inp_var : inputs) {
      if (inp_var->PreOp() && !inp_var->IsStopGradient()) {
        VLOG(3) << "add pre op " << inp_var->PreOp()->Type() << " in slot "
                << inp_name;
        pre_ops_list.emplace_back(inp_var->PreOp());
        pre_ops_out_idx_list.push_back(inp_var->PreOpOutIdx());
      } else {
        VLOG(3) << "no pre op in slot " << inp_name
                << " input var stop_gradient: " << inp_var->IsStopGradient();
        pre_ops_list.emplace_back(nullptr);
        // pre_ops_out_idx_list.push_back(-1);
      }
    }
  }

  std::string type_;
  int trace_id_;

  // Note: each fwd op corresponds to a vector of bwd ops.
  std::vector<framework::OpDesc*> grad_op_descs_;

  platform::Place place_;

  OpBasePtrMap pre_ops_;
  std::map<std::string, std::vector<int>> pre_ops_out_idx_;

  VarBaseWeakPtrList outputs_ref;
  // Inputs to a vector of bwd ops.
  std::vector<VarBasePtrMap> grad_input_vars_;
  // Outputs to a vector of bwd ops.
  std::vector<VarBasePtrMap> grad_output_vars_;

  std::vector<py::object> backward_hooks_;

  framework::AttributeMap attrs_;
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<std::shared_ptr<VarBase>> Forward(
      const std::vector<std::shared_ptr<VarBase>>& inputs) {
    std::vector<std::shared_ptr<VarBase>> vars;
    return vars;
  }
};

// infer var type context for imperative mode
class PYBIND11_HIDDEN RuntimeInferVarTypeContext
    : public framework::InferVarTypeContext {
 public:
  RuntimeInferVarTypeContext(const imperative::VarBasePtrMap* inputs,
                             imperative::VarBasePtrMap* outputs,
                             const framework::AttributeMap* attrs_map)
      : InferVarTypeContext(nullptr, nullptr),
        inputs_(inputs),
        outputs_(outputs),
        attrs_(attrs_map),
        input_names_(),
        output_names_(),
        var_set_() {
    input_names_.reserve(inputs_->size());
    for (auto& it : *inputs_) {
      for (std::shared_ptr<imperative::VarBase> var : it.second) {
        input_names_[it.first].emplace_back(var->Name());
        var_set_[var->Name()] = var;
      }
    }

    output_names_.reserve(outputs_->size());
    for (auto& it : *outputs_) {
      for (std::shared_ptr<imperative::VarBase> var : it.second) {
        output_names_[it.first].emplace_back(var->Name());
        var_set_[var->Name()] = var;
      }
    }
  }

  virtual ~RuntimeInferVarTypeContext() {}

  framework::Attribute GetAttr(const std::string& name) const override {
    PADDLE_ENFORCE_NOT_NULL(attrs_);
    return attrs_->at(name);
  }

  bool HasVar(const std::string& name) const override {
    return var_set_.count(name) > 0;
  }

  bool HasInput(const std::string& name) const override {
    PADDLE_ENFORCE_NOT_NULL(inputs_);
    return inputs_->count(name) > 0;
  }

  bool HasOutput(const std::string& name) const override {
    PADDLE_ENFORCE_NOT_NULL(outputs_);
    return outputs_->count(name) > 0;
  }

  const std::vector<std::string>& Input(
      const std::string& name) const override {
    return input_names_.at(name);
  }

  const std::vector<std::string>& Output(
      const std::string& name) const override {
    return output_names_.at(name);
  }

  framework::proto::VarType::Type GetType(
      const std::string& name) const override {
    return var_set_.at(name)->Type();
  }

  void SetType(const std::string& name,
               framework::proto::VarType::Type type) override {
    if (name == "kLookupTablePath") {
      VLOG(2) << "SUPER UGLY FIX, remove this when move imperative mode in C++";
    } else {
      var_set_[name]->SetType(type);
    }
  }

  framework::proto::VarType::Type GetDataType(
      const std::string& name) const override {
    return var_set_.at(name)->DataType();
  }

  void SetDataType(const std::string& name,
                   framework::proto::VarType::Type type) override {
    var_set_[name]->SetDataType(type);
  }

  std::vector<framework::proto::VarType::Type> GetDataTypes(
      const std::string& name) const override {
    PADDLE_THROW("GetDataTypes is not supported in runtime InferVarType");
  }

  void SetDataTypes(const std::string& name,
                    const std::vector<framework::proto::VarType::Type>&
                        multiple_data_type) override {
    PADDLE_THROW("SetDataTypes is not supported in runtime InferVarType");
  }

  std::vector<int64_t> GetShape(const std::string& name) const override {
    PADDLE_THROW("Do not handle Shape in runtime InferVarType");
  }

  void SetShape(const std::string& name,
                const std::vector<int64_t>& dims) override {
    PADDLE_THROW("Do not handle Shape in runtime InferVarType");
  }

  int32_t GetLoDLevel(const std::string& name) const override {
    PADDLE_THROW("Do not handle LoDLevel in runtime InferVarType");
  }

  void SetLoDLevel(const std::string& name, int32_t lod_level) override {
    PADDLE_THROW("Do not handle LoDLevel in runtime InferVarType");
  }

 private:
  const imperative::VarBasePtrMap* inputs_;
  imperative::VarBasePtrMap* outputs_;
  const framework::AttributeMap* attrs_;
  std::unordered_map<std::string, std::vector<std::string>> input_names_;
  std::unordered_map<std::string, std::vector<std::string>> output_names_;
  std::unordered_map<std::string, std::shared_ptr<imperative::VarBase>>
      var_set_;
};

}  // namespace imperative
}  // namespace paddle
