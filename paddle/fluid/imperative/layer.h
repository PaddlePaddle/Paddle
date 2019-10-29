// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <list>
#include <map>     // NOLINT
#include <memory>  // NOLINT
#include <mutex>   // NOLINT
#include <set>
#include <string>         // NOLINT
#include <unordered_map>  // NOLINT
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/flags.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

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

class VarBase {
  DISABLE_COPY_AND_ASSIGN(VarBase);

 public:
  static std::vector<std::string> AliveVarNames();
  explicit VarBase(bool has_grad, const std::string& name)
      : name_(name),
        grad_var_(has_grad ? new VarBase(false, GradVarName()) : nullptr) {
    if (IsDebugEnabled()) {
      VLOG(10) << "Construct VarBase: " << name;
      name_set_.Insert(name_);
    }
  }

  explicit VarBase(const std::string& name) : VarBase(true, name) {}

  ~VarBase() {
    VLOG(10) << "Destruct VarBase: " << name_;
    if (IsDebugEnabled()) {
      name_set_.Remove(name_);
    }
  }

  const framework::Variable& Var() const { return var_; }

  framework::Variable* MutableVar() { return &var_; }

  bool HasGradVar() const { return grad_var_ != nullptr; }

  const std::shared_ptr<VarBase>& GradVarBase() const { return grad_var_; }

  const framework::Variable& GradVar() const {
    PADDLE_ENFORCE_NOT_NULL(grad_var_, "Gradient of %s does not exist", name_);
    return grad_var_->var_;
  }

  framework::Variable* MutableGradVar() {
    PADDLE_ENFORCE_NOT_NULL(grad_var_, "Gradient of %s does not exist", name_);
    return &(grad_var_->var_);
  }

  // This is used for python api
  void SetOverridedStopGradient(bool stop_gradient) {
    if (stop_gradient) {
      overrided_stop_gradient_ = 1;
    } else {
      overrided_stop_gradient_ = 0;
    }
    if (grad_var_) {
      grad_var_->SetOverridedStopGradient(stop_gradient);
    }
  }
  // This is used for python api
  bool OverridedStopGradient() const {
    if (overrided_stop_gradient_ == 0) {
      return false;
    } else {
      return true;
    }
  }

  // This is used inside C++
  int InnerOverridedStopGradient() const { return overrided_stop_gradient_; }

  bool GradGenerated() const { return grad_generated_; }

  void SetGradGenerated(bool generated) { grad_generated_ = generated; }
  // This is used inside C++
  void InnerSetOverridedStopGradient(bool stop_gradient) {
    if (overrided_stop_gradient_ == -1) {
      overrided_stop_gradient_ = static_cast<int>(stop_gradient);
      if (grad_var_) {
        grad_var_->InnerSetOverridedStopGradient(stop_gradient);
      }
    } else {
      VLOG(6) << "Ignore Stop gradient conversion for Var: " << Name()
              << "Set value is: " << overrided_stop_gradient_;
    }
  }

  void SetPersistable(bool persistable) { persistable_ = persistable; }

  bool Persistable() const { return persistable_; }

  void AddGradOps(const std::weak_ptr<OpBase>& op);

  std::vector<OpBase*> GradOps() {
    std::vector<OpBase*> rlt;
    // TODO(jiabin): use better data structure to remove nullptr when we find it
    for (const auto& wk_ptr : grad_ops_) {
      OpBase* tmp_op = wk_ptr.lock().get();
      if (tmp_op) rlt.emplace_back(tmp_op);
    }
    return rlt;
  }
  void ClearGradOps() { grad_ops_.clear(); }

  const std::string& Name() const { return name_; }

  void SetName(const std::string& name) {
    name_ = name;
    if (grad_var_) {
      grad_var_->SetName(GradVarName());
    }
  }

  std::string GradVarName() { return framework::GradVarName(name_); }

  void SetType(framework::proto::VarType::Type type) { type_ = type; }

  framework::proto::VarType::Type Type() const { return type_; }

  void SetDataType(framework::proto::VarType::Type data_type) {
    data_type_ = data_type;
    if (grad_var_) {
      grad_var_->SetDataType(data_type_);
    }
  }

  framework::proto::VarType::Type DataType() const { return data_type_; }

  void ClearGradient();

  std::shared_ptr<VarBase> NewVarBase(const platform::Place& dst_place,
                                      const bool blocking) const;

 private:
  framework::Variable var_;
  std::string name_;
  std::shared_ptr<VarBase> grad_var_;
  mutable size_t copied_counter_ = 0;

  // grad_op indicates which grad_op will this var be used as input
  std::vector<std::weak_ptr<OpBase>> grad_ops_;
  // add this property for users may set stop_gradient themselves and this
  // should override the
  // frameworks setting (-1) unset, (1) true, (0) false
  int overrided_stop_gradient_{-1};
  bool grad_generated_{false};
  bool persistable_{false};

  framework::proto::VarType::Type type_{framework::proto::VarType::LOD_TENSOR};
  framework::proto::VarType::Type data_type_{framework::proto::VarType::FP32};
  static ThreadSafeNameSet name_set_;
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<std::shared_ptr<VarBase>> Forward(
      const std::vector<std::shared_ptr<VarBase>>& inputs) {
    return {};
  }
};

// infer var type context for imperative mode
class RuntimeInferVarTypeContext : public framework::InferVarTypeContext {
 public:
  RuntimeInferVarTypeContext(const NameVarBaseMap& inputs,
                             const NameVarBaseMap* outputs,
                             const framework::AttributeMap& attrs_map)
      : InferVarTypeContext(nullptr, nullptr),
        inputs_(inputs),
        outputs_(outputs),
        attrs_(attrs_map),
        input_names_(),
        output_names_(),
        var_set_() {
    input_names_.reserve(inputs_.size());
    for (auto& it : inputs_) {
      for (auto& var : it.second) {
        input_names_[it.first].emplace_back(var->Name());
        var_set_[var->Name()] = var.get();
      }
    }

    output_names_.reserve(outputs_->size());
    for (auto& it : *outputs_) {
      for (auto& var : it.second) {
        output_names_[it.first].emplace_back(var->Name());
        var_set_[var->Name()] = var.get();
      }
    }
  }

  virtual ~RuntimeInferVarTypeContext() {}

  framework::Attribute GetAttr(const std::string& name) const override {
    auto iter = attrs_.find(name);
    PADDLE_ENFORCE_EQ(iter != attrs_.end(), true, "Cannot find attribute %s",
                      name);
    return iter->second;
  }

  bool HasVar(const std::string& name) const override {
    return var_set_.count(name) > 0;
  }

  bool HasInput(const std::string& name) const override {
    return inputs_.count(name) > 0;
  }

  bool HasOutput(const std::string& name) const override {
    PADDLE_ENFORCE_NOT_NULL(outputs_);
    return outputs_->count(name) > 0;
  }

  const std::vector<std::string>& Input(
      const std::string& name) const override {
    auto iter = input_names_.find(name);
    PADDLE_ENFORCE_EQ(iter != input_names_.end(), true, "Cannot find input %s",
                      name);
    return iter->second;
  }

  const std::vector<std::string>& Output(
      const std::string& name) const override {
    auto iter = output_names_.find(name);
    PADDLE_ENFORCE_EQ(iter != output_names_.end(), true,
                      "Cannot find output %s", name);
    return iter->second;
  }

  framework::proto::VarType::Type GetType(
      const std::string& name) const override {
    auto iter = var_set_.find(name);
    PADDLE_ENFORCE_EQ(iter != var_set_.end(), true,
                      "Cannot find var %s in GetType", name);
    return iter->second->Type();
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
    auto iter = var_set_.find(name);
    PADDLE_ENFORCE_EQ(iter != var_set_.end(), true,
                      "Cannot find var %s in GetDataType", name);
    return iter->second->DataType();
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
  const NameVarBaseMap& inputs_;
  const NameVarBaseMap* outputs_;
  const framework::AttributeMap& attrs_;
  std::unordered_map<std::string, std::vector<std::string>> input_names_;
  std::unordered_map<std::string, std::vector<std::string>> output_names_;
  std::unordered_map<std::string, VarBase*> var_set_;
};

// TODO(zjl): to support py_func layer
class OpBase : public std::enable_shared_from_this<OpBase> {
  DISABLE_COPY_AND_ASSIGN(OpBase);

 public:
  ~OpBase() { VLOG(3) << "Destruct Op: " << Type() << std::endl; }

  // Developer should not rely on this method to create OpBase.
  // OpBase should be created in Tracer and managed by Tracer totally.
  template <typename... Args>
  static std::shared_ptr<OpBase> Create(Args&&... args) {
    return std::shared_ptr<OpBase>(new OpBase(std::forward<Args>(args)...));
  }

  size_t id() const { return id_; }

  const std::string& Type() const { return op_->Type(); }

  void Run(const NameVarBaseMap& ins, const NameVarBaseMap& outs);

  const framework::VariableNameMap& InputNameMap() const {
    return op_->Inputs();
  }

  const framework::VariableNameMap& OutputNameMap() const {
    return op_->Outputs();
  }

  const framework::AttributeMap& Attrs() const { return op_->Attrs(); }
  const framework::OpInfo& Info() const { return op_->Info(); }

  void ClearBackwardTrace();

  const std::vector<OpBase*>& GradPendingOps() const {
    return grad_pending_ops_;
  }

  void InsertGradPendingOps(OpBase* op) { grad_pending_ops_.emplace_back(op); }

  void SortGradPendingOps() {
    std::sort(grad_pending_ops_.begin(), grad_pending_ops_.end(),
              [](OpBase* op1, OpBase* op2) { return op1->id() > op2->id(); });
  }
  NameVarBaseMap* GetMutableOutsMap() { return &outs_; }
  NameVarBaseMap* GetMutableInsMap() { return &ins_; }
  const NameVarBaseMap& GetInsMap() { return ins_; }
  const NameVarBaseMap& GetOutsMap() { return outs_; }
  const platform::Place& place() const { return place_; }

  // TODO(jiabin) prepare for backward hook
  void RegisterBackwardHooks(const std::function<void()>& func) {
    backward_hooks_.emplace_back(func);
  }

  void InvokeBackwardHooks() {
    for (const auto& func : backward_hooks_) {
      func();
      VLOG(5) << "Invoke Backward Hook for: " << Type() << std::endl;
    }
  }

 private:
  OpBase(size_t id, const std::string& type, const NameVarBaseMap& ins,
         const NameVarBaseMap& outs, framework::AttributeMap attrs,
         const platform::Place& place);

  OpBase(size_t id, const framework::OpDesc& op_desc,
         const platform::Place& place);

  size_t id_;

  std::unique_ptr<framework::OperatorBase> op_;

  std::vector<std::function<void()>> backward_hooks_;
  platform::Place place_;

  // Not need to be std::weak_ptr, because op is binded to a certain Tracer,
  // and would not be used by a Tracer that does not create itself.
  std::vector<OpBase*> grad_pending_ops_;

  // This part is only used for backward
  NameVarBaseMap ins_;
  NameVarBaseMap outs_;
};

}  // namespace imperative
}  // namespace paddle
