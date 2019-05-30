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
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

class VarBase;
class OpBase;
class Tracer;

class VarBase {
  DISABLE_COPY_AND_ASSIGN(VarBase);

 public:
  explicit VarBase(bool has_grad)
      : grad_var_(has_grad ? new VarBase(false) : nullptr) {}

  VarBase() : VarBase(true) {}

  ~VarBase() { VLOG(10) << "Destruct variable " << name_; }

  const framework::Variable& Var() const { return var_; }

  framework::Variable* MutableVar() { return &var_; }

  bool HasGradVar() const { return grad_var_ != nullptr; }

  const std::shared_ptr<VarBase>& GradVarBase() const { return grad_var_; }

  const framework::Variable& GradVar() const {
    PADDLE_ENFORCE_NOT_NULL(grad_var_, "Gradient of %s does not exist");
    return grad_var_->var_;
  }

  framework::Variable* MutableGradVar() {
    PADDLE_ENFORCE_NOT_NULL(grad_var_, "Gradient of %s does not exist");
    return &(grad_var_->var_);
  }

  void SetStopGradient(bool stop_gradient) {
    stop_gradient_ = stop_gradient;
    if (grad_var_) {
      grad_var_->stop_gradient_ = stop_gradient;
    }
  }

  bool StopGradient() const { return stop_gradient_; }

  void SetPersistable(bool persistable) { persistable_ = persistable; }

  bool Persistable() const { return persistable_; }

  void SetGeneratedOp(OpBase* op);

  OpBase* GeneratedOp() { return generated_op_.lock().get(); }

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
  }

  framework::proto::VarType::Type DataType() const { return data_type_; }

  void ClearGradient();

  std::shared_ptr<VarBase> NewVarBase(const platform::Place& dst_place,
                                      const bool blocking) const;

 private:
  framework::Variable var_;
  std::shared_ptr<VarBase> grad_var_;

  std::weak_ptr<OpBase> generated_op_;

  std::string name_;
  bool stop_gradient_{false};
  bool persistable_{false};

  framework::proto::VarType::Type type_{framework::proto::VarType::LOD_TENSOR};
  framework::proto::VarType::Type data_type_{framework::proto::VarType::FP32};
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<std::shared_ptr<VarBase>> Forward(
      const std::vector<std::shared_ptr<VarBase>>& inputs) {
    return {};
  }
};

using NameVarBaseMap =
    std::map<std::string, std::vector<std::shared_ptr<VarBase>>>;

// TODO(zjl): to support py_func layer
class OpBase : public std::enable_shared_from_this<OpBase> {
  DISABLE_COPY_AND_ASSIGN(OpBase);

 public:
  virtual ~OpBase() = default;

  // Developer should not rely on this method to create OpBase.
  // OpBase should be created in Tracer and managed by Tracer totally.
  template <typename... Args>
  static std::shared_ptr<OpBase> Create(Args&&... args) {
    return std::shared_ptr<OpBase>(new OpBase(std::forward<Args>(args)...));
  }

  size_t id() const { return id_; }

  const std::string& Type() const { return type_; }

  void ClearBackwardTrace();

  const std::vector<std::unique_ptr<framework::OpDesc>>& GradOpDescs() const {
    return grad_op_descs_;
  }

  const std::unordered_set<OpBase*>& PrecedingOps() const {
    return preceding_ops_;
  }

  const std::vector<NameVarBaseMap>& BackwardInputs() const { return bwd_ins_; }

  const std::vector<NameVarBaseMap>& BackwardOutputs() const {
    return bwd_outs_;
  }

  Tracer* tracer() { return tracer_; }

  const platform::Place& place() const { return place_; }

 private:
  OpBase(Tracer* tracer, size_t id, const std::string& type,
         const NameVarBaseMap& ins, const NameVarBaseMap& outs,
         framework::AttributeMap attrs, const platform::Place& place,
         bool trace_backward);

  OpBase(Tracer* tracer, size_t id, const framework::OpDesc& op_desc,
         const NameVarBaseMap& ins, const NameVarBaseMap& outs,
         const platform::Place& place, bool trace_backward);

  void TraceBackward(const framework::OpDesc& fwd_op, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs);

  void RunOp(const std::unique_ptr<framework::OperatorBase>& op,
             const NameVarBaseMap& ins, const NameVarBaseMap& outs);

  Tracer* tracer_;
  size_t id_;
  std::string type_;
  const framework::OpInfo* info_;

  platform::Place place_;

  // Not need to be std::weak_ptr, because op is binded to a certain Tracer,
  // and would not be used by a Tracer that does not create itself.
  std::unordered_set<OpBase*> preceding_ops_;

  std::vector<std::unique_ptr<framework::OpDesc>> grad_op_descs_;
  std::vector<NameVarBaseMap> bwd_ins_;
  std::vector<NameVarBaseMap> bwd_outs_;
};

class Tracer {
  DISABLE_COPY_AND_ASSIGN(Tracer);

 public:
  explicit Tracer(const framework::BlockDesc* block_desc)
      : block_desc_(block_desc) {}

  ~Tracer() = default;

  void TraceOp(const std::string& type, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, framework::AttributeMap attrs,
               const platform::Place& place, bool trace_bacward);

  void TraceOp(const framework::OpDesc& op_desc, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, const platform::Place& place,
               bool trace_backward);

  void RemoveOp(size_t id) {
    PADDLE_ENFORCE(ops_.erase(id) > 0, "Op with id %d is not inside tracer",
                   id);
  }

  void RemoveOp(OpBase* op) {
    PADDLE_ENFORCE_NOT_NULL(op, "Cannot remove null op");
    auto iter = ops_.find(op->id());
    PADDLE_ENFORCE(iter != ops_.end() && iter->second.get() == op,
                   "Op is not inside tracer");
    ops_.erase(iter);
  }

  void Clear() { ops_.clear(); }

 private:
  static size_t GenerateUniqueId() {
    static std::atomic<size_t> id{0};
    return id.fetch_add(1);
  }

 private:
  std::unordered_map<size_t, std::shared_ptr<OpBase>> ops_;
  const framework::BlockDesc* block_desc_;
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
    PADDLE_ENFORCE(iter != attrs_.end(), "Cannot find attribute %s", name);
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
    PADDLE_ENFORCE(iter != input_names_.end(), "Cannot find input %s", name);
    return iter->second;
  }

  const std::vector<std::string>& Output(
      const std::string& name) const override {
    auto iter = output_names_.find(name);
    PADDLE_ENFORCE(iter != output_names_.end(), "Cannot find output %s", name);
    return iter->second;
  }

  framework::proto::VarType::Type GetType(
      const std::string& name) const override {
    auto iter = var_set_.find(name);
    PADDLE_ENFORCE(iter != var_set_.end(), "Cannot find var %s in GetType",
                   name);
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
    PADDLE_ENFORCE(iter != var_set_.end(), "Cannot find var %s in GetDataType",
                   name);
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

void AutoGrad(VarBase* var, const detail::BackwardStrategy& strategy);

}  // namespace imperative
}  // namespace paddle
