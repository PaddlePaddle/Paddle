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
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

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
    PADDLE_ENFORCE_NOT_NULL(grad_var_, "Gradient of %s does not exist", name_);
    return grad_var_->var_;
  }

  framework::Variable* MutableGradVar() {
    PADDLE_ENFORCE_NOT_NULL(grad_var_, "Gradient of %s does not exist", name_);
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

  const std::string& Type() const { return op_->Type(); }

  void Run(const NameVarBaseMap& ins, const NameVarBaseMap& outs);

  const framework::VariableNameMap& InputNameMap() const {
    return op_->Inputs();
  }

  const framework::VariableNameMap& OutputNameMap() const {
    return op_->Outputs();
  }

  const framework::AttributeMap& Attrs() const { return op_->Attrs(); }

  void ClearBackwardTrace();

  const std::vector<std::unique_ptr<framework::OpDesc>>& GradOpDescs() const {
    return grad_op_descs_;
  }

  const std::vector<OpBase*>& PrecedingOps() const { return preceding_ops_; }

  const std::vector<NameVarBaseMap>& BackwardInputs() const { return bwd_ins_; }

  const std::vector<NameVarBaseMap>& BackwardOutputs() const {
    return bwd_outs_;
  }

  Tracer* HoldedTracer() { return holded_tracer_; }

  const platform::Place& place() const { return place_; }

  void TraceBackward(const framework::OpDesc& fwd_op, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs);

 private:
  OpBase(Tracer* tracer, size_t id, const std::string& type,
         const NameVarBaseMap& ins, const NameVarBaseMap& outs,
         framework::AttributeMap attrs, const platform::Place& place);

  OpBase(Tracer* tracer, size_t id, const framework::OpDesc& op_desc,
         const NameVarBaseMap& ins, const NameVarBaseMap& outs,
         const platform::Place& place);

  Tracer* holded_tracer_;
  size_t id_;

  std::unique_ptr<framework::OperatorBase> op_;

  platform::Place place_;

  // Not need to be std::weak_ptr, because op is binded to a certain Tracer,
  // and would not be used by a Tracer that does not create itself.
  std::vector<OpBase*> preceding_ops_;

  std::vector<std::unique_ptr<framework::OpDesc>> grad_op_descs_;
  std::vector<NameVarBaseMap> bwd_ins_;
  std::vector<NameVarBaseMap> bwd_outs_;
};

}  // namespace imperative
}  // namespace paddle
