// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/event.h"

namespace paddle {
namespace framework {

using OpKernelComputeFunc = std::function<void(const ExecutionContext&)>;
using OpKernelMap =
    std::unordered_map<OpKernelType, OpKernelComputeFunc, OpKernelType::Hash>;

class InterpretercoreInferShapeContext : public InferShapeContext {
 public:
  InterpretercoreInferShapeContext(const OperatorBase& op,
                                   const RuntimeContext& ctx)
      : op_(op), ctx_(ctx), can_skip_lod_(false) {}

  bool HasInput(const std::string& name) const override {
    // has only one input
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end()) {
      return false;
    }
    const auto& in = it->second;
    if (in.size() == 0) return false;
    PADDLE_ENFORCE_EQ(
        in.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input %s should not contain more than one inputs.", name));
    return in[0] != nullptr;
  }

  bool HasOutput(const std::string& name) const override {
    // has only one output
    const auto& outs = ctx_.outputs;
    auto it = outs.find(name);
    if (it == outs.end()) {
      return false;
    }
    const auto& out = it->second;
    if (out.size() == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(
        out.size(), 1UL,
        platform::errors::InvalidArgument(
            "Output %s should not contain more than one outputs.", name));
    return out[0] != nullptr;
  }

  bool HasInputs(const std::string& name) const override {
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end() || it->second.empty()) {
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
    const auto& outs = ctx_.outputs;
    auto it = outs.find(name);
    if (it == outs.end() || it->second.empty()) {
      return false;
    }
    for (auto& output : it->second) {
      if (output == nullptr) {
        return false;
      }
    }
    return true;
  }

  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

  std::vector<std::string> Inputs(const std::string& name) const override {
    return op_.Inputs(name);
  }

  std::vector<std::string> Outputs(const std::string& name) const override {
    return op_.Outputs(name);
  }

  std::string GetInputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(idx, op_proto->inputs().size(),
                      platform::errors::OutOfRange(
                          "The index should be less than the size of inputs of "
                          "operator %s, but got index is %d and size is %d",
                          op_.Type(), idx, op_proto->inputs().size()));
    return op_proto->inputs()[idx].name();
  }

  std::string GetOutputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(
        idx, op_proto->outputs().size(),
        platform::errors::OutOfRange(
            "The index should be less than the size of outputs of "
            "operator %s, but got index is %d and size is %d",
            op_.Type(), idx, op_proto->outputs().size()));
    return op_proto->outputs()[idx].name();
  }

  void ShareDim(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.inputs.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, in_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          in_it->second.size(), i));
    PADDLE_ENFORCE_LT(j, out_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          out_it->second.size(), j));

    Variable* in_var = in_it->second[i];
    Variable* out_var = out_it->second[j];

    PADDLE_ENFORCE_EQ(
        in_var->Type(), out_var->Type(),
        platform::errors::InvalidArgument(
            "The type of input (%s) and output (%s) are inconsistent.", in,
            out));

    if (in_var->IsType<framework::SelectedRows>()) {
      auto& in_sele_rows = in_var->Get<framework::SelectedRows>();
      auto out_sele_rows = out_var->GetMutable<framework::SelectedRows>();
      out_sele_rows->mutable_value()->Resize(in_sele_rows.value().dims());
      out_sele_rows->set_rows(in_sele_rows.rows());
      out_sele_rows->set_height(in_sele_rows.height());
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto& in_lod_tensor = in_var->Get<framework::LoDTensor>();
      auto* out_lod_tensor = out_var->GetMutable<framework::LoDTensor>();
      out_lod_tensor->Resize(in_lod_tensor.dims());
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, the input type of ShareDim only can be LoDTensor "
          "or SelectedRows."));
    }
  }

  void ShareAllLoD(const std::string& in,
                   const std::string& out) const override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(in_it, ctx_.inputs.end(),
                      platform::errors::NotFound(
                          "Input [%s] found error in Op [%s]", in, op_.Type()));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output [%s] found error in Op [%s]", out,
                                   op_.Type()));

    auto& in_var_list = in_it->second;
    auto& out_var_list = out_it->second;

    PADDLE_ENFORCE_EQ(
        in_var_list.size(), out_var_list.size(),
        platform::errors::PreconditionNotMet(
            "Op [%s]: Input var size should be equal with output var size",
            op_.Type()));

    auto& out_var_names = op_.Outputs(out);

    for (size_t i = 0; i < in_var_list.size(); ++i) {
      if (out_var_names[i] == framework::kEmptyVarName) {
        continue;
      }

      Variable* in_var = in_var_list[i];
      if (!in_var->IsType<LoDTensor>()) return;
      Variable* out_var = out_var_list[i];
      PADDLE_ENFORCE_EQ(out_var->IsType<LoDTensor>(), true,
                        platform::errors::PreconditionNotMet(
                            "The %d-th output of Output(%s) must be LoDTensor.",
                            i, out_var_names[i]));
      auto& in_tensor = in_var->Get<LoDTensor>();
      auto* out_tensor = out_var->GetMutable<LoDTensor>();
      out_tensor->set_lod(in_tensor.lod());
#ifdef PADDLE_WITH_MKLDNN
      if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
        out_tensor->set_layout(in_tensor.layout());
    }
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    if (can_skip_lod_) {
      return;
    }
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.inputs.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, in_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          in_it->second.size(), i));
    PADDLE_ENFORCE_LT(j, out_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          out_it->second.size(), j));

    Variable* in_var = in_it->second.at(i);
    if (!in_var->IsType<LoDTensor>()) return;
    Variable* out_var = out_it->second.at(j);
    PADDLE_ENFORCE_EQ(
        out_var->IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The %zu-th output of Output(%s) must be LoDTensor.", j, out));
    auto& in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());

// TODO(dzhwinter) : reuse ShareLoD in most operators.
// Need to call ShareLayout explicitly in sequence related ops.
// Shall we have a better method to shared info between in/out Tensor?
#ifdef PADDLE_WITH_MKLDNN
    // Fix me: ugly workaround below
    // Correct solution:
    //    set_layout() should NOT be called here (i.e. ShareLoD). Instead,
    //    layout of output tensor should be set "manually" in Compute()
    //    of each OPKernel. The reason layout should NOT be shared between
    //    input and output "automatically" (now by InferShape()->ShareLoD())
    //    is that layout transform may occur after InferShape().
    // Workaround:
    //    Skip set_layout() when input layout is kMKLDNN
    //    This is to avoid kMKLDNN is populated wrongly into a non-MKLDNN
    //    OPKernel. In all MKLDNN OPkernel, set_layout(kMKLDNN) should be called
    //    in Compute()
    if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
      out_tensor->set_layout(in_tensor.layout());
  }

  int32_t GetLoDLevel(const std::string& in, size_t i = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }

  void SetLoDLevel(const std::string& out, int32_t lod_level,
                   size_t j = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }

  bool IsRuntime() const override { return true; }

  // TODO(paddle-dev): Can this be template?
  std::vector<InferShapeVarPtr> GetInputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = InputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }

  std::vector<InferShapeVarPtr> GetOutputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = OutputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }

  DDim GetInputDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(%s) should hold one element, but now it holds %zu elements.",
            name, vars.size()));
    return this->GetDim(vars[0]);
  }

  std::vector<DDim> GetInputsDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    return GetDims(vars);
  }

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string& name) const override {
    return GetVarTypes(InputVars(name));
  }

  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string& name) const override {
    return GetVarTypes(OutputVars(name));
  }

  void SetOutputDim(const std::string& name, const DDim& dim) override {
    auto& vars = OutputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument("Output(%s) should hold one element, "
                                          "but now it holds %zu elements.",
                                          name, vars.size()));
    SetDim(vars[0], dim);
  }

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override {
    auto& vars = OutputVars(name);
    SetDims(vars, dims);
  }

  void SetSkipLoD(bool skip) { can_skip_lod_ = skip; }

 protected:
  DDim GetDim(Variable* var) const {
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Input variable is nullptr."));
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Only LoDTensor or SelectedRows support 'GetDim', but input "
          "Variable's type is %s.",
          ToTypeName(var->Type())));
    }
  }

  std::vector<DDim> GetDims(const std::vector<Variable*>& vars) const {
    std::vector<DDim> ret;
    ret.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(ret),
                   [this](Variable* var) { return this->GetDim(var); });
    return ret;
  }

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetRepeatedDims method only ban be used in compile time."));
  }

  void SetDim(Variable* var, const DDim& dim) {
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Variable type error, expect LoDTensor or SelectedRows, but received "
          "(%s).",
          ToTypeName(var->Type())));
    }
  }

  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims) {
    size_t length = vars.size();
    PADDLE_ENFORCE_EQ(length, dims.size(),
                      platform::errors::InvalidArgument(
                          "The number of input variables do not match the "
                          "number of input dimensions, the number of variables "
                          "is %zu, the number of dimensions is %zu.",
                          length, dims.size()));
    for (size_t i = 0; i < length; ++i) {
      if (vars[i] == nullptr) {
        continue;
      }
      SetDim(vars[i], dims[i]);
    }
  }

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetRepeatedDims method only can be used in compile time."));
  }

  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<Variable*>& vars) const {
    std::vector<proto::VarType::Type> retv;
    retv.resize(vars.size());
    std::transform(
        vars.begin(), vars.end(), retv.begin(),
        std::bind(std::mem_fn(&InterpretercoreInferShapeContext::GetVarType),
                  this, std::placeholders::_1));
    return retv;
  }

  proto::VarType::Type GetVarType(Variable* var) const {
    return ToVarType(var->Type());
  }

 private:
  const std::vector<Variable*>& InputVars(const std::string& name) const {
    auto it = ctx_.inputs.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.inputs.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the input (%s).", op_.Type(), name));
    return it->second;
  }

  const std::vector<Variable*>& OutputVars(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.outputs.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the outputs (%s).", op_.Type(), name));
    return it->second;
  }

  const OperatorBase& op_;
  const RuntimeContext& ctx_;
  bool can_skip_lod_;
};

struct OpKernelFunc {
  OpKernelComputeFunc compute_func_;
};

struct VariableMetaInfo {
  int var_ref_count_;
  paddle::framework::VarDesc* vardesc_;
};

// TODO(zhiqiu): Maybe we need to add rwlock for VariableScope?

// NOTE(xiongkun03): Use scope as a member of VariableScope, we don't need
// ScopeBase. Scope manager the variables and VariableScope is just a quick
// access machanism. ScopeListener is the callback to sync changes in Original
// Scope. We can make it a membership of VariableScope. Here we use inherent.
class VariableScope : public ScopeBase, public ScopeListener {
 public:
  VariableScope(Scope* outer_scope) {
    // for @EMPTY@ variable
    var_list_.push_back(nullptr);
    name2id_[kEmptyVarName] = 0;
    VariableMetaInfo info;
    info.var_ref_count_ = 0;
    info.vardesc_ = nullptr;
    vec_meta_info_.push_back(info);
    outer_scope_ = outer_scope;

    PADDLE_ENFORCE_NE(
        outer_scope_, nullptr,
        platform::errors::PreconditionNotMet(
            "You have passed a nullptr to construct VariableScope."));
    outer_scope->AddListener(this);
  }

  ~VariableScope() {
    if (outer_scope_ != nullptr) outer_scope_->DelListener(this);
  }

  const Scope* GetScope() const { return outer_scope_; }

  Variable* FindVar(const std::string& name) const {
    auto it = name2id_.find(name);
    if (it != name2id_.end()) {
      PADDLE_ENFORCE_LT(it->second, var_list_.size(),
                        platform::errors::NotFound(
                            "The id(%d) of variable(%s) should not be larger "
                            "than the size of variable list(%d).",
                            it->second, name, var_list_.size()));
      return var_list_[it->second];
    }
    return nullptr;
  }

  // Get variable id by name, return -1 if not found
  int GetIdByName(const std::string& name) const {
    auto it = name2id_.find(name);
    if (it != name2id_.end()) {
      return it->second;
    }
    return -1;
  }

  // Get variable name by id, return "" if not found
  std::string GetNameById(int id) const {
    // NOTE(zhiqiu): do not use vec_meta_info_[id].vardesc_->Name() since
    // vec_meta_info_[id] may be nullptr,
    // typically when the target variable is not existed in the original program
    // desc, but created by interpretercore.
    // For example, created and used by d2h_copy or h2d_copy operator.
    auto it =
        std::find_if(name2id_.begin(), name2id_.end(),
                     [id](const auto& pair) { return pair.second == id; });
    if (it != name2id_.end()) {
      return it->first;
    }
    return "";
  }

  bool HasVar(const std::string& name) const {
    return name2id_.find(name) != name2id_.end();
  }

  int VarId(const std::string& name) const {
    CheckExist(name);
    return name2id_.at(name);
  }

  Variable* Var(int id) const { return var_list_.at(id); }

  Variable* Var(const std::string& name) const {
    return var_list_.at(VarId(name));
  }

  size_t VarSize() const { return var_list_.size(); }

  void AddVar(const std::string& name, VarDesc* var_desc) {  // NOLINT
    // AddVar -> Scope::Var -> onCreateVariable.
    VLOG(4) << "Add variable: " << name << " through AddVar()";
    auto v = outer_scope_->Var(name);
    if (nullptr == var_desc) {
      v->GetMutable<LoDTensor>();
    } else {
      InitializeVariable(
          v,
          var_desc
              ->GetType());  // Scope don't initialize variable recently created
    }
    SetVarDesc(name, var_desc);
  }

  void AddVar(const std::string& name, Variable& var) {  // NOLINT
    // Though name existed in outer_scope_, we need
    // add again to create name2id map.
    outer_scope_->Var(name);
  }

  void SetVarDesc(const std::string& name, framework::VarDesc* var_desc) {
    CheckExist(name);
    vec_meta_info_[VarId(name)].vardesc_ = var_desc;
  }

  paddle::framework::VarDesc* VarDesc(const std::string& name) const {
    return VarDesc(VarId(name));
  }

  paddle::framework::VarDesc* VarDesc(int id) const {
    CheckExist(id);
    return vec_meta_info_[id].vardesc_;
  }

  void CheckExist(int id) const {
    PADDLE_ENFORCE_LT(id, var_list_.size(),
                      platform::errors::PreconditionNotMet(
                          "Required var_id < %d, but received var_id = %d.",
                          var_list_.size(), id));
  }

  void CheckExist(const std::string& name) const {
    PADDLE_ENFORCE_EQ(
        HasVar(name), true,
        platform::errors::NotFound("%s not in VariableScope.", name));
  }

 public:  // callbacks from ScopeListener class
  void onCreateVariable(const std::string& name) override {
    auto v = outer_scope_->GetVar(name);  // must exsit in outer_scope_
    if (!HasVar(name)) {                  // may exist in variable scope.
      VLOG(4) << "Calling VariableScope::onCreateVariable with var_name: "
              << name;
      name2id_[name] = VarSize();
      var_list_.push_back(v);

      VariableMetaInfo info;
      info.var_ref_count_ = 0;
      info.vardesc_ = nullptr;  // set nullptr, then modifty it in AddVar()
      vec_meta_info_.push_back(info);
    }
  }
  void onDeleteVariable(const std::string& name) override {
    if (HasVar(name)) {
      VLOG(4) << "Calling VariableScope::onDeleteVariable with var_name: "
              << name;
    }
  }
  void onRenameVariable(const std::string& old_name,
                        const std::string& new_name) override {}
  void onCreateScope(Scope* Scope) override {}
  void onDeleteScope(Scope* Scope) override {}
  void onClear() override {}
  std::vector<VariableMetaInfo>& MutableVecMetaInfo() { return vec_meta_info_; }

  const std::vector<VariableMetaInfo>& VecMetaInfo() const {
    return vec_meta_info_;
  }

 private:
  std::vector<Variable*> var_list_;
  std::map<std::string, int> name2id_;
  std::vector<VariableMetaInfo> vec_meta_info_;
  Scope* outer_scope_ = nullptr;
};

class NextInstruction {
 public:
  void AddDirectRun(size_t id) { direct_run_.push_back(id); }

  void ADDEventRun(size_t id) { event_wait_run_.push_back(id); }

  void AddSyncRun(size_t id) { synchronize_run_.push_back(id); }

  const std::vector<size_t>& DirectRunIds() const { return direct_run_; }

  const std::vector<size_t>& EventRunIds() const { return event_wait_run_; }

  const std::vector<size_t>& SyncRunIds() const { return synchronize_run_; }

 private:
  std::vector<size_t> direct_run_;
  std::vector<size_t> event_wait_run_;
  std::vector<size_t> synchronize_run_;
};

struct EventInter {
  explicit EventInter(size_t var_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type)
      : var_id_(var_id), event_(event), waiter_type_(waiter_type) {}
  size_t var_id_;
  std::shared_ptr<platform::DeviceEvent> event_;
  platform::DeviceType waiter_type_;
};

struct InstructionInfo {
  std::vector<size_t> dependecy_count_;
};

enum class OpFuncType {
  kQueueSync = 0,   // CPU kernel, block host
  kQueueAsync = 1,  // GPU Kernel or d2h, h2d, send, recv, broadcast
};
class RuntimeInferShapeContext;

struct OpFuncNode {
  OperatorBase* operator_base_;
  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;
  std::unordered_set<int> no_data_transform_index;

  OpKernelComputeFunc kernel_func_;
  platform::DeviceContext* dev_ctx_;  // not owned
  OpFuncType type_;
};

class Instruction {
 public:
  Instruction(size_t id, const OpFuncNode& op_func_node,
              const platform::DeviceContext& dev_ctx)
      : id_(id), op_func_node_(op_func_node), dev_ctx_(dev_ctx) {
    PADDLE_ENFORCE_GE(id, 0, platform::errors::PreconditionNotMet(
                                 "Required id >= 0, but received id = %d", id));
  }

  size_t Id() const { return id_; }

  const std::map<std::string, std::vector<int>>& Inputs() const {
    return op_func_node_.input_index;
  }

  const std::map<std::string, std::vector<int>>& Outputs() const {
    return op_func_node_.output_index;
  }

  const std::unordered_set<int>& NoDataTransformVars() const {
    return op_func_node_.no_data_transform_index;
  }

  OpKernelComputeFunc KernelFunc() const { return op_func_node_.kernel_func_; }

  OpFuncType KernelType() const { return op_func_node_.type_; }

  OperatorBase* OpBase() const {
    auto* op_base = op_func_node_.operator_base_;
    PADDLE_ENFORCE_NOT_NULL(op_base, platform::errors::PreconditionNotMet(
                                         "op_base shall not be nullptr."));
    return op_base;
  }

  NextInstruction& NextInstructions() { return next_instruction_; }

  const NextInstruction& NextInstructions() const { return next_instruction_; }

  void AddGCCheckVar(size_t id) { gc_check_var_list_.push_back(id); }

  const std::vector<size_t>& GCCheckVars() const { return gc_check_var_list_; }

  void ResetContext(const VariableValueMap& in_vars,
                    const VariableValueMap& out_vars) {
    runtime_ctx_.reset(new RuntimeContext(in_vars, out_vars));
    infershape_ctx_.reset(
        new InterpretercoreInferShapeContext(*OpBase(), *runtime_ctx_.get()));
    // NOTE: Because execution_ctx_ is constructed by `scope&`, so we fake an
    // empty here to avoid illegal local reference.
    static framework::Scope scope_;
    execution_ctx_.reset(
        new ExecutionContext(*OpBase(), scope_, dev_ctx_, *runtime_ctx_.get()));
  }

  std::shared_ptr<RuntimeContext> InnerRuntimeContext() const {
    return runtime_ctx_;
  }

  std::shared_ptr<InterpretercoreInferShapeContext> InnerInferShapeContext()
      const {
    return infershape_ctx_;
  }

  std::shared_ptr<ExecutionContext> InnerExecutionContext() const {
    return execution_ctx_;
  }

  const platform::DeviceContext& DeviceContext() const { return dev_ctx_; }

  const std::vector<std::pair<Variable*, Variable*>>& InplaceInfo() const {
    return vec_inplace_in_to_out_;
  }

  void AddInplace(Variable* in, Variable* out) {
    vec_inplace_in_to_out_.emplace_back(in, out);
  }

  const std::vector<EventInter>& InputEvents() const { return intput_events_; }

  const std::vector<EventInter>& OutputEvents() const { return output_events_; }

  void AddInputEvent(size_t var_id,
                     std::shared_ptr<platform::DeviceEvent> event,
                     platform::DeviceType waiter_type) {
    intput_events_.emplace_back(var_id, event, waiter_type);
  }

  void AddOutputEvent(size_t var_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type) {
    output_events_.emplace_back(var_id, event, waiter_type);
  }

 private:
  size_t id_;
  const OpFuncNode& op_func_node_;          // not owned
  const platform::DeviceContext& dev_ctx_;  // not owned

  std::shared_ptr<RuntimeContext> runtime_ctx_;
  std::shared_ptr<InterpretercoreInferShapeContext> infershape_ctx_;
  std::shared_ptr<ExecutionContext> execution_ctx_;

  std::vector<size_t> gc_check_var_list_;
  NextInstruction next_instruction_;

  std::vector<EventInter> intput_events_;
  std::vector<EventInter> output_events_;

  std::vector<std::pair<Variable*, Variable*>> vec_inplace_in_to_out_;
};

namespace interpreter {
static constexpr char kMemcpyH2D[] = "memcpy_h2d";
static constexpr char kMemcpyD2H[] = "memcpy_d2h";

static bool IsMemcpyH2D(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyH2D;
}

static bool IsMemcpyD2H(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyD2H;
}
}  // namespace interpreter

}  // namespace framework
}  // namespace paddle
