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
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/flags.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/variable_wrapper.h"
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
      : var_(std::make_shared<VariableWrapper>(name)),
        grad_var_(has_grad ? new VarBase(false, GradVarName()) : nullptr) {
    if (IsDebugEnabled()) {
      VLOG(10) << "Construct VarBase: " << Name();
      name_set_.Insert(Name());
    }
  }

  explicit VarBase(const std::string& name) : VarBase(true, name) {}

  ~VarBase() {
    VLOG(10) << "Destruct VarBase: " << Name();
    if (IsDebugEnabled()) {
      name_set_.Remove(Name());
    }
  }

  const std::shared_ptr<VariableWrapper>& SharedVar() const { return var_; }

  const framework::Variable& Var() const { return var_->Var(); }

  framework::Variable* MutableVar() { return var_->MutableVar(); }

  bool HasGradVar() const { return grad_var_ != nullptr; }

  const std::shared_ptr<VarBase>& GradVarBase() const { return grad_var_; }

  void ClearGradVarBase() { grad_var_ = nullptr; }

  const std::shared_ptr<VarBase>& MutableGradVarBase() {
    if (grad_var_ == nullptr) {
      grad_var_ = std::make_shared<VarBase>(false, GradVarName());
      // NOTE(zhiqiu): we should keep grad_var_'s stop_gradient property same as
      // fwd varbase
      grad_var_->SetOverridedStopGradient(var_->InnerOverridedStopGradient());
    }
    return grad_var_;
  }

  const framework::Variable& GradVar() const {
    PADDLE_ENFORCE_NOT_NULL(
        grad_var_,
        platform::errors::NotFound("Gradient of %s does not exist", Name()));
    return grad_var_->Var();
  }

  framework::Variable* MutableGradVar() {
    PADDLE_ENFORCE_NOT_NULL(
        grad_var_,
        platform::errors::NotFound("Gradient of %s does not exist", Name()));
    return grad_var_->MutableVar();
  }

  void SetOverridedStopGradient(bool stop_gradient) {
    var_->SetOverridedStopGradient(stop_gradient);
    if (grad_var_) {
      grad_var_->SetOverridedStopGradient(stop_gradient);
    }
  }

  bool OverridedStopGradient() const { return var_->OverridedStopGradient(); }

  void InnerSetOverridedStopGradient(bool stop_gradient) {
    if (var_->InnerOverridedStopGradient() == -1) {
      var_->InnerSetOverridedStopGradient(stop_gradient);
      if (grad_var_) {
        grad_var_->InnerSetOverridedStopGradient(stop_gradient);
      }
    }
  }

  void SetPersistable(bool persistable) { var_->SetPersistable(persistable); }

  bool Persistable() const { return var_->Persistable(); }

  // Only grad var is allowed to call these 2 methods
  void AddGradOp(const std::shared_ptr<OpBase>& op) {
    if (op &&
        std::find(grad_ops_.begin(), grad_ops_.end(), op) == grad_ops_.end()) {
      grad_ops_.emplace_back(op);
    }
  }

  const std::vector<std::shared_ptr<OpBase>>& GradOps() const {
    return grad_ops_;
  }

  void ClearGradOps() { grad_ops_.clear(); }

  const std::string& Name() const { return var_->Name(); }

  void SetName(const std::string& name) {
    var_->SetName(name);
    if (grad_var_) {
      grad_var_->SetName(GradVarName());
    }
  }

  std::string GradVarName() { return framework::GradVarName(Name()); }

  void SetType(framework::proto::VarType::Type type) { var_->SetType(type); }

  framework::proto::VarType::Type Type() const { return var_->Type(); }

  void SetDataType(framework::proto::VarType::Type data_type) {
    var_->SetDataType(data_type);
    if (grad_var_) {
      grad_var_->SetDataType(data_type);
    }
  }

  framework::proto::VarType::Type DataType() const { return var_->DataType(); }

  void ClearGradient();

  std::shared_ptr<VarBase> NewVarBase(const platform::Place& dst_place,
                                      const bool blocking) const;

 private:
  /**
   * NOTE(zengjinle): never remove the const qualifier of `var_` if you are
   * not very familiar with the autograd idea (including the higher order
   * derivative).
   */
  const std::shared_ptr<VariableWrapper> var_;

  std::shared_ptr<VarBase> grad_var_;
  std::vector<std::shared_ptr<OpBase>> grad_ops_;

  mutable size_t copied_counter_ = 0;

  static ThreadSafeNameSet name_set_;
};

using VariableWrapperList = std::vector<std::shared_ptr<VariableWrapper>>;

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<std::shared_ptr<VarBase>> Forward(
      const std::vector<std::shared_ptr<VarBase>>& inputs) {
    return {};
  }
};

template <typename VarType>
class DygraphExecutionContext : public framework::ExecutionContext {
  using Variable = framework::Variable;

 public:
  DygraphExecutionContext(const framework::OperatorBase& op,
                          const framework::Scope& scope,
                          const platform::DeviceContext& device_context,
                          const framework::RuntimeContext& ctx,
                          std::vector<framework::KernelConfig>* configs,
                          const NameVarMap<VarType>& var_base_map_in,
                          const NameVarMap<VarType>& var_base_map_out,
                          const framework::AttributeMap& attrs)
      : ExecutionContext(op, scope, device_context, ctx, configs),
        var_base_map_in_(var_base_map_in),
        var_base_map_out_(var_base_map_out),
        attrs_(attrs) {}

  std::string InputName(const std::string& name) const override {
    auto it = var_base_map_in_.find(name);
    PADDLE_ENFORCE_NE(it, var_base_map_in_.end(),
                      platform::errors::PreconditionNotMet(
                          "Can not find [%s] in Input", name));
    return it->second[0]->Name();
  }
  std::vector<std::string> InputNames(const std::string& name) const override {
    auto it = var_base_map_in_.find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_in_.end(),
        platform::errors::NotFound("Can not find [%s] in Input", name));
    std::vector<std::string> vec_res;
    vec_res.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) {
      vec_res.push_back(it->second[i]->Name());
    }
    return vec_res;
  }

  std::string OutputName(const std::string& name) const override {
    auto it = var_base_map_out_.find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_out_.end(),
        platform::errors::NotFound("Can not find [%s] in Output", name));
    return it->second[0]->Name();
  }

  std::vector<std::string> OutputNames(const std::string& name) const override {
    auto it = var_base_map_out_.find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_out_.end(),
        platform::errors::NotFound("Can not find [%s] in Output", name));
    std::vector<std::string> vec_res;
    vec_res.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) {
      vec_res.push_back(it->second[i]->Name());
    }
    return vec_res;
  }

  bool HasAttr(const std::string& name) const override {
    return attrs_.count(name) != 0;
  }

  const framework::AttributeMap& Attrs() const override { return attrs_; }

  const framework::Attribute& GetAttr(const std::string& name) const override {
    auto it = attrs_.find(name);

    PADDLE_ENFORCE_NE(
        it, attrs_.end(),
        platform::errors::NotFound("can not find [%s] in attrs", name));

    return it->second;
  }

  std::vector<std::string> InNameList() const override {
    std::vector<std::string> vec_temp;
    vec_temp.reserve(var_base_map_in_.size());

    for (auto& v : var_base_map_in_) {
      vec_temp.push_back(v.first);
    }

    return vec_temp;
  }
  bool HasInput(const std::string& name) const override {
    auto it = var_base_map_in_.find(name);
    return (it != var_base_map_in_.end() && it->second.size() > 0);
  }

  bool HasOutput(const std::string& name) const override {
    auto it = var_base_map_out_.find(name);
    return (it != var_base_map_out_.end() && it->second.size() > 0);
  }

  size_t InputSize(const std::string& name) const override {
    return InputNames(name).size();
  }

  size_t OutputSize(const std::string& name) const override {
    return OutputNames(name).size();
  }

  const Variable* InputVar(const std::string& name) const override {
    auto it = var_base_map_in_.find(name);
    if (it == var_base_map_in_.end()) {
      return nullptr;
    }

    return it->second.empty() ? nullptr : it->second[0]->MutableVar();
  }

  Variable* OutputVar(const std::string& name) const override {
    auto it = var_base_map_out_.find(name);
    if (it == var_base_map_out_.end()) {
      return nullptr;
    }

    return it->second.empty() ? nullptr : it->second[0]->MutableVar();
  }

  const std::vector<Variable*> MultiInputVar(
      const std::string& name) const override {
    auto it = var_base_map_in_.find(name);
    if (it == var_base_map_in_.end()) {
      return {};
    }
    std::vector<Variable*> vec_res;
    vec_res.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) {
      vec_res.push_back(it->second[i]->MutableVar());
    }

    return vec_res;
  }

  std::vector<Variable*> MultiOutputVar(
      const std::string& name) const override {
    auto it = var_base_map_out_.find(name);
    if (it == var_base_map_out_.end()) {
      return {};
    }
    std::vector<Variable*> vec_res;
    vec_res.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) {
      vec_res.push_back(it->second[i]->MutableVar());
    }

    return vec_res;
  }

 private:
  const NameVarMap<VarType>& var_base_map_in_;
  const NameVarMap<VarType>& var_base_map_out_;
  const framework::AttributeMap& attrs_;
};

// infer var type context for imperative mode
template <typename VarType>
class RuntimeInferVarTypeContext : public framework::InferVarTypeContext {
 public:
  RuntimeInferVarTypeContext(const NameVarMap<VarType>& inputs,
                             const NameVarMap<VarType>* outputs,
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
    auto it = inputs_.find(name);
    return (it != inputs_.end() && it->second.size() > 0);
  }

  bool HasOutput(const std::string& name) const override {
    PADDLE_ENFORCE_NOT_NULL(outputs_);
    auto it = outputs_->find(name);
    return (it != outputs_->end() && it->second.size() > 0);
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
      if ((var_set_[name]->MutableVar()->IsInitialized() == true) &&
          (var_set_[name]->MutableVar()->Type() != type)) {
        var_set_[name]->MutableVar()->Clear();
      }
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
  const NameVarMap<VarType>& inputs_;
  const NameVarMap<VarType>* outputs_;
  const framework::AttributeMap& attrs_;
  std::unordered_map<std::string, std::vector<std::string>> input_names_;
  std::unordered_map<std::string, std::vector<std::string>> output_names_;
  std::unordered_map<std::string, VarType*> var_set_;
};

// TODO(zjl): to support py_func layer
class OpBase {
  DISABLE_COPY_AND_ASSIGN(OpBase);

 public:
  OpBase() = default;

  ~OpBase() { VLOG(3) << "Destruct Op: " << Type(); }

  size_t id() const { return id_; }

  const std::string& Type() const { return op_->Type(); }

  const framework::AttributeMap& Attrs() const { return attrs_; }

  const framework::OpInfo& Info() const { return op_->Info(); }

  const framework::OperatorBase& InnerOp() const { return *op_; }

  void ClearBackwardTrace();

  const std::vector<std::shared_ptr<OpBase>>& GradPendingOps() const {
    return grad_pending_ops_;
  }

  void SetGradPendingOps(std::vector<std::shared_ptr<OpBase>> pending_ops) {
    grad_pending_ops_ = std::move(pending_ops);
  }

  NameVarMap<VariableWrapper>* GetMutableOutsMap() { return &outs_; }

  NameVarMap<VariableWrapper>* GetMutableInsMap() { return &ins_; }

  const NameVarMap<VariableWrapper>& GetInsMap() { return ins_; }

  const NameVarMap<VariableWrapper>& GetOutsMap() { return outs_; }

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

  void SetType(const std::string& type);

  void CheckAttrs() {
    auto& info = op_->Info();
    if (info.Checker() != nullptr) {
      info.Checker()->Check(&attrs_, true);
    }
  }

  void SetInput(const std::string& name, VariableWrapperList vars) {
    ins_[name] = std::move(vars);
  }

  void SetOutput(const std::string& name, VariableWrapperList vars) {
    outs_[name] = std::move(vars);
  }

  void SetAttrMap(const framework::AttributeMap& attrs) { attrs_ = attrs; }

  void SetAttr(const std::string& name, const framework::Attribute& v) {
    attrs_[name] = v;
  }

  void SetBlockAttr(const std::string& name, framework::BlockDesc* block) {
    PADDLE_THROW("SetBlockAttr is not support in dygraph OpBase");
  }

  const framework::AttributeMap& Attrs() { return attrs_; }

  void SetId(size_t id) { id_ = id; }

  void SetPlace(const platform::Place& place) { place_ = place; }

  bool HasAttr(const std::string& name) const { return attrs_.count(name) > 0; }

  const framework::Attribute& GetAttr(const std::string& name) const {
    auto it = attrs_.find(name);
    PADDLE_ENFORCE(it != attrs_.end(), "can not find attribute [%s]", name);

    return it->second;
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return boost::get<T>(GetAttr(name));
  }

  void AddAllowedEmptyVar(const VariableWrapper* var) {
    allow_empty_vars_.emplace(var);
  }

  bool IsAllowedEmptyVar(const VariableWrapper* var) {
    return allow_empty_vars_.count(var) > 0;
  }

  static void Run(const framework::OperatorBase& op,
                  const NameVarMap<VarBase>& ins,
                  const NameVarMap<VarBase>& outs,
                  const framework::AttributeMap& attrs,
                  const platform::Place& place);

  static void Run(const framework::OperatorBase& op,
                  const NameVarMap<VariableWrapper>& ins,
                  const NameVarMap<VariableWrapper>& outs,
                  const framework::AttributeMap& attrs,
                  const platform::Place& place);

 private:
  NameVarMap<VariableWrapper> ins_;
  NameVarMap<VariableWrapper> outs_;
  framework::AttributeMap attrs_;
  std::unique_ptr<framework::OperatorBase> op_;

  std::vector<std::shared_ptr<OpBase>> grad_pending_ops_;
  platform::Place place_;

  std::unordered_set<const VariableWrapper*> allow_empty_vars_;

  size_t id_{-1UL};

  std::vector<std::function<void()>> backward_hooks_;
};

template <typename VarType>
class DygraphInferShapeContext : public framework::InferShapeContext {
  using DDim = framework::DDim;

 public:
  DygraphInferShapeContext(const NameVarMap<VarType>* in,
                           const NameVarMap<VarType>* out,
                           const framework::AttributeMap* attr)
      : var_base_map_in_(in), var_base_map_out_(out), attrs_(attr) {}

  bool HasInput(const std::string& name) const override {
    // has only one input
    auto it = var_base_map_in_->find(name);

    if (it == var_base_map_in_->end()) {
      return false;
    }
    const auto& in = it->second;
    if (in.size() == 0) return false;
    PADDLE_ENFORCE_EQ(
        in.size(), 1UL,
        platform::errors::PreconditionNotMet(
            "Input %s should not have more than one inputs", name));
    return in[0] != nullptr;
  }

  bool HasOutput(const std::string& name) const override {
    // has only one output
    auto it = var_base_map_out_->find(name);
    if (it == var_base_map_out_->end()) {
      return false;
    }
    const auto& out = it->second;
    if (out.size() == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(
        out.size(), 1UL,
        platform::errors::PreconditionNotMet(
            "Output %s should not have more than one outputs", name));
    return out[0] != nullptr;
  }

  bool HasInputs(const std::string& name) const override {
    auto it = var_base_map_in_->find(name);
    if (it == var_base_map_in_->end() || it->second.empty()) {
      return false;
    }
    for (auto& input : it->second) {
      if (input == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool HasOutputs(const std::string& name) const override {
    auto it = var_base_map_out_->find(name);
    if (it == var_base_map_out_->end() || it->second.empty()) {
      return false;
    }
    for (auto& output : it->second) {
      if (output == nullptr) {
        return false;
      }
    }
    return true;
  }

  framework::AttrReader Attrs() const override {
    return framework::AttrReader(*attrs_);
  }

  std::vector<std::string> Inputs(const std::string& name) const override {
    // return op_.Inputs(name);
    std::vector<std::string> vec_res;
    auto it = var_base_map_in_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_in_->end(),
        platform::errors::NotFound("can not find [%s] in input", name));

    vec_res.reserve(it->second.size());
    for (auto& var : it->second) {
      vec_res.push_back(var->Name());
    }

    return vec_res;
  }

  std::vector<std::string> Outputs(const std::string& name) const override {
    std::vector<std::string> vec_res;
    auto it = var_base_map_out_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_out_->end(),
        platform::errors::NotFound("can not find [%s] in output", name));

    vec_res.reserve(it->second.size());
    for (auto& var : it->second) {
      vec_res.push_back(var->Name());
    }

    return vec_res;
  }

  void ShareDim(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) override {
    auto in_it = var_base_map_in_->find(in);
    auto out_it = var_base_map_out_->find(out);
    PADDLE_ENFORCE_NE(
        in_it, var_base_map_in_->end(),
        platform::errors::NotFound("can not found [%s] in input", in));
    PADDLE_ENFORCE_GT(in_it->second.size(), i,
                      platform::errors::PreconditionNotMet(
                          "Inputs %s should have %llu argument", in, i));
    PADDLE_ENFORCE_NE(
        out_it, var_base_map_out_->end(),
        platform::errors::NotFound("can not found [%s] in input", in));
    PADDLE_ENFORCE_GT(out_it->second.size(), j,
                      platform::errors::PreconditionNotMet(
                          "Outputs %s should have %llu argument", out, j));

    framework::Variable* in_var = in_it->second[i]->MutableVar();
    framework::Variable* out_var = out_it->second[j]->MutableVar();

    PADDLE_ENFORCE_EQ(in_var->Type(), out_var->Type(),
                      platform::errors::PreconditionNotMet(
                          "The type of %s and %s is not the same.", in, out));

    if (in_var->IsType<framework::LoDTensor>()) {
      auto& in_lod_tensor = in_var->Get<framework::LoDTensor>();
      auto* out_lod_tensor = out_var->GetMutable<framework::LoDTensor>();
      out_lod_tensor->Resize(in_lod_tensor.dims());
    } else {
      auto& in_sele_rows = in_var->Get<framework::SelectedRows>();
      auto out_sele_rows = out_var->GetMutable<framework::SelectedRows>();
      out_sele_rows->mutable_value()->Resize(in_sele_rows.value().dims());
      out_sele_rows->set_rows(in_sele_rows.rows());
      out_sele_rows->set_height(in_sele_rows.height());
    }
  }

  void ShareAllLoD(const std::string& in,
                   const std::string& out) const override {
    // do nothing
  }
  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    // do nothing
  }

  bool IsRuntime() const override { return true; }

  // TODO(paddle-dev): Can this be template?
  std::vector<framework::InferShapeVarPtr> GetInputVarPtrs(
      const std::string& name) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "GetInputVarPtrs not support in dygraph runtime context"));
  }

  std::vector<framework::InferShapeVarPtr> GetOutputVarPtrs(
      const std::string& name) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "GetOutputVarPtrs not support in dygraph runtime context"));
  }

  DDim GetInputDim(const std::string& name) const override {
    auto it = var_base_map_in_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_in_->end(),
        platform::errors::NotFound("can not find [%s] in input", name));
    PADDLE_ENFORCE_EQ(
        it->second.size(), 1UL,
        platform::errors::PreconditionNotMet(
            "Input(%s) should hold one element, but now it holds %d", name,
            it->second.size()));
    return this->GetDim(it->second[0]->MutableVar());
  }

  std::vector<DDim> GetInputsDim(const std::string& name) const override {
    // const std::vector<Variable*>& vars = InputVars(name);
    std::vector<DDim> vec_res;
    auto it = var_base_map_in_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_in_->end(),
        platform::errors::NotFound("can not find [%s] in output", name));
    vec_res.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) {
      vec_res.emplace_back(GetDim(it->second[i]->MutableVar()));
    }

    return vec_res;
  }

  std::vector<framework::proto::VarType::Type> GetInputsVarType(
      const std::string& name) const override {
    std::vector<framework::proto::VarType::Type> vec_res;
    auto it = var_base_map_in_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_in_->end(),
        platform::errors::NotFound("can not find [%s] in input", name));
    vec_res.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) {
      vec_res.emplace_back(
          framework::ToVarType(it->second[i]->MutableVar()->Type()));
    }
    return vec_res;
  }

  std::vector<framework::proto::VarType::Type> GetOutputsVarType(
      const std::string& name) const override {
    std::vector<framework::proto::VarType::Type> vec_res;
    auto it = var_base_map_out_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_out_->end(),
        platform::errors::NotFound("can not find [%s] in output", name));
    vec_res.reserve(it->second.size());
    for (size_t i = 0; i < it->second.size(); ++i) {
      vec_res.emplace_back(
          framework::ToVarType(it->second[i]->MutableVar()->Type()));
    }
    return vec_res;
  }

  void SetOutputDim(const std::string& name, const DDim& dim) override {
    auto it = var_base_map_out_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_out_->end(),
        platform::errors::NotFound("can not find [%s] in output", name));

    SetDim(it->second[0]->MutableVar(), dim);
  }

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override {
    auto it = var_base_map_out_->find(name);
    PADDLE_ENFORCE_NE(
        it, var_base_map_out_->end(),
        platform::errors::NotFound("can not find [%s] in output", name));

    PADDLE_ENFORCE_EQ(it->second.size(), dims.size(),
                      platform::errors::PreconditionNotMet(
                          "dim size [%d] is not match output var number [%d]",
                          dims.size(), it->second.size()));

    for (size_t i = 0; i < dims.size(); ++i) {
      SetDim(it->second[i]->MutableVar(), dims[i]);
    }
  }

  int32_t GetLoDLevel(const std::string& in, size_t i = 0) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "GetLoDLevel function not support in dygraph mode"));
  }

  void SetLoDLevel(const std::string& out, int32_t lod_level,
                   size_t j = 0) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "SetLoDLevel function not support in dygraph mode"));
  }

 protected:
  DDim GetDim(framework::Variable* var) const {
    PADDLE_ENFORCE_NOT_NULL(var, platform::errors::PreconditionNotMet(
                                     "Input variable should not be null"));
    if (var->IsType<framework::LoDTensor>()) {
      return var->Get<framework::LoDTensor>().dims();
    } else if (var->IsType<framework::SelectedRows>()) {
      return var->Get<framework::SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Only LoDTensor/SelectedRows support 'GetDim', but Variables "
          "type_id is xx."));
    }
  }

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "GetRepeatedDims not support in dygraph runtime"));
  }

  void SetDim(framework::Variable* var, const DDim& dim) {
    if (var->IsType<framework::LoDTensor>()) {
      var->GetMutable<framework::LoDTensor>()->Resize(dim);
    } else if (var->IsType<framework::SelectedRows>()) {
      var->GetMutable<framework::SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Variable type_id %s, expect LoDTensor/SelectedRows."));
    }
  }

  void SetDims(const std::vector<framework::Variable*>& vars,
               const std::vector<DDim>& dims) {
    size_t length = vars.size();
    PADDLE_ENFORCE_EQ(
        length, dims.size(),
        platform::errors::PreconditionNotMet(
            "Vars number [%d] should be equal with dims number [%d]", length,
            dims.size()));
    for (size_t i = 0; i < length; ++i) {
      if (vars[i] == nullptr) {
        continue;
      }
      SetDim(vars[i], dims[i]);
    }
  }

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "SetRepeatedDims not support in dygraph runtime"));
  }

 private:
  const NameVarMap<VarType>* var_base_map_in_;
  const NameVarMap<VarType>* var_base_map_out_;
  const framework::AttributeMap* attrs_;
};

}  // namespace imperative
}  // namespace paddle
