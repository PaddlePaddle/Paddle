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

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/core/utils/rw_lock.h"

#define SCOPE_VARS_READER_LOCK AutoRDLock auto_lock(&vars_lock_);
#define SCOPE_VARS_WRITER_LOCK AutoWRLock auto_lock(&vars_lock_);

namespace paddle {
namespace framework {

InterpretercoreInferShapeContext::InterpretercoreInferShapeContext(
    const OperatorBase& op, const RuntimeContext& ctx)
    : op_(op), ctx_(ctx), can_skip_lod_(false) {}

bool InterpretercoreInferShapeContext::HasInput(const std::string& name) const {
  // has only one input
  const auto& ins = ctx_.inputs;
  auto it = ins.find(name);
  if (it == ins.end()) {
    return false;
  }
  const auto& in = it->second;
  if (in.size() == 0) return false;
  PADDLE_ENFORCE_EQ(
      in.size(),
      1UL,
      platform::errors::InvalidArgument(
          "Input %s should not contain more than one inputs.", name));
  return in[0] != nullptr;
}

bool InterpretercoreInferShapeContext::HasOutput(
    const std::string& name) const {
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
      out.size(),
      1UL,
      platform::errors::InvalidArgument(
          "Output %s should not contain more than one outputs.", name));
  return out[0] != nullptr;
}

bool InterpretercoreInferShapeContext::HasAttr(const std::string& name) const {
  return op_.HasAttr(name);
}

bool InterpretercoreInferShapeContext::HasInputs(
    const std::string& name) const {
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

bool InterpretercoreInferShapeContext::HasOutputs(const std::string& name,
                                                  bool allow_null) const {
  const auto& outs = ctx_.outputs;
  auto it = outs.find(name);
  if (it == outs.end() || it->second.empty()) {
    return false;
  }
  if (allow_null) {
    for (auto& output : it->second) {
      if (output != nullptr) return true;
    }
    return false;
  } else {
    for (auto& output : it->second) {
      if (output == nullptr) return false;
    }
    return true;
  }
}

AttrReader InterpretercoreInferShapeContext::Attrs() const {
  return AttrReader(op_.Attrs(), op_.RuntimeAttrs());
}

std::vector<std::string> InterpretercoreInferShapeContext::Inputs(
    const std::string& name) const {
  return op_.Inputs(name);
}

std::vector<std::string> InterpretercoreInferShapeContext::Outputs(
    const std::string& name) const {
  return op_.Outputs(name);
}

std::string InterpretercoreInferShapeContext::GetInputNameByIdx(
    size_t idx) const {
  auto& op_proto =
      paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
  PADDLE_ENFORCE_LT(idx,
                    op_proto->inputs().size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of inputs of "
                        "operator %s, but got index is %d and size is %d",
                        op_.Type(),
                        idx,
                        op_proto->inputs().size()));
  return op_proto->inputs()[idx].name();
}

std::string InterpretercoreInferShapeContext::GetOutputNameByIdx(
    size_t idx) const {
  auto& op_proto =
      paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
  PADDLE_ENFORCE_LT(idx,
                    op_proto->outputs().size(),
                    platform::errors::OutOfRange(
                        "The index should be less than the size of outputs of "
                        "operator %s, but got index is %d and size is %d",
                        op_.Type(),
                        idx,
                        op_proto->outputs().size()));
  return op_proto->outputs()[idx].name();
}

void InterpretercoreInferShapeContext::ShareDim(const std::string& in,
                                                const std::string& out,
                                                size_t i,
                                                size_t j) {
  auto in_it = ctx_.inputs.find(in);
  auto out_it = ctx_.outputs.find(out);
  PADDLE_ENFORCE_NE(in_it,
                    ctx_.inputs.end(),
                    platform::errors::NotFound("Input %s does not exist.", in));
  PADDLE_ENFORCE_NE(
      out_it,
      ctx_.outputs.end(),
      platform::errors::NotFound("Output %s does not exist.", out));
  PADDLE_ENFORCE_LT(i,
                    in_it->second.size(),
                    platform::errors::InvalidArgument(
                        "The index of input dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        in_it->second.size(),
                        i));
  PADDLE_ENFORCE_LT(j,
                    out_it->second.size(),
                    platform::errors::InvalidArgument(
                        "The index of output dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        out_it->second.size(),
                        j));

  Variable* in_var = in_it->second[i];
  Variable* out_var = out_it->second[j];

  PADDLE_ENFORCE_EQ(
      in_var->Type(),
      out_var->Type(),
      platform::errors::InvalidArgument(
          "The type of input (%s) and output (%s) are inconsistent.", in, out));

  if (in_var->IsType<phi::SelectedRows>()) {
    auto& in_sele_rows = in_var->Get<phi::SelectedRows>();
    auto out_sele_rows = out_var->GetMutable<phi::SelectedRows>();
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

void InterpretercoreInferShapeContext::ShareAllLoD(
    const std::string& in, const std::string& out) const {
  auto in_it = ctx_.inputs.find(in);
  auto out_it = ctx_.outputs.find(out);
  PADDLE_ENFORCE_NE(in_it,
                    ctx_.inputs.end(),
                    platform::errors::NotFound(
                        "Input [%s] found error in Op [%s]", in, op_.Type()));
  PADDLE_ENFORCE_NE(out_it,
                    ctx_.outputs.end(),
                    platform::errors::NotFound(
                        "Output [%s] found error in Op [%s]", out, op_.Type()));

  auto& in_var_list = in_it->second;
  auto& out_var_list = out_it->second;

  PADDLE_ENFORCE_EQ(
      in_var_list.size(),
      out_var_list.size(),
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
    PADDLE_ENFORCE_EQ(out_var->IsType<LoDTensor>(),
                      true,
                      platform::errors::PreconditionNotMet(
                          "The %d-th output of Output(%s) must be LoDTensor.",
                          i,
                          out_var_names[i]));
    auto& in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());
#ifdef PADDLE_WITH_MKLDNN
    if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
      out_tensor->set_layout(in_tensor.layout());
  }
}

void InterpretercoreInferShapeContext::ShareLoD(const std::string& in,
                                                const std::string& out,
                                                size_t i,
                                                size_t j) const {
  if (can_skip_lod_) {
    return;
  }
  auto in_it = ctx_.inputs.find(in);
  auto out_it = ctx_.outputs.find(out);
  PADDLE_ENFORCE_NE(in_it,
                    ctx_.inputs.end(),
                    platform::errors::NotFound("Input %s does not exist.", in));
  PADDLE_ENFORCE_NE(
      out_it,
      ctx_.outputs.end(),
      platform::errors::NotFound("Output %s does not exist.", out));
  PADDLE_ENFORCE_LT(i,
                    in_it->second.size(),
                    platform::errors::InvalidArgument(
                        "The index of input dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        in_it->second.size(),
                        i));
  PADDLE_ENFORCE_LT(j,
                    out_it->second.size(),
                    platform::errors::InvalidArgument(
                        "The index of output dimension is out of range, "
                        "excepted index less than %zu, but received %zu.",
                        out_it->second.size(),
                        j));

  Variable* in_var = in_it->second.at(i);
  if (!in_var->IsType<LoDTensor>()) return;
  Variable* out_var = out_it->second.at(j);
  PADDLE_ENFORCE_EQ(
      out_var->IsType<LoDTensor>(),
      true,
      platform::errors::InvalidArgument(
          "The %zu-th output of Output(%s) must be LoDTensor.", j, out));
  auto& in_tensor = in_var->Get<LoDTensor>();
  auto* out_tensor = out_var->GetMutable<LoDTensor>();
  out_tensor->set_lod(in_tensor.lod());

// TODO(dzhwinter) : reuse ShareLoD in most operators.
// Need to call ShareLayout explicitly in sequence related ops.
// Shall we have a better method to shared info between in/out phi::DenseTensor?
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

int32_t InterpretercoreInferShapeContext::GetLoDLevel(const std::string& in,
                                                      size_t i) const {
  PADDLE_THROW(platform::errors::PreconditionNotMet(
      "GetLoDLevel is only used in compile time. The calculation of "
      "output's actual lod is different among operators so that should be "
      "set in the runtime kernel."));
}

void InterpretercoreInferShapeContext::SetLoDLevel(const std::string& out,
                                                   int32_t lod_level,
                                                   size_t j) const {
  PADDLE_THROW(platform::errors::PreconditionNotMet(
      "SetLoDLevel is only used in compile time. The calculation of "
      "output's actual lod is different among operators so that should be "
      "set in the runtime kernel."));
}

bool InterpretercoreInferShapeContext::IsRuntime() const { return true; }

bool InterpretercoreInferShapeContext::IsRunMKLDNNKernel() const {
  try {
    auto& op_with_kernel = dynamic_cast<const OperatorWithKernel&>(op_);
    return ((op_with_kernel.kernel_type()) &&
            (op_with_kernel.kernel_type()->data_layout_ ==
             framework::DataLayout::kMKLDNN));
  } catch (std::bad_cast& exp) {
    return false;
  }
}

// TODO(paddle-dev): Can this be template?
paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize>
InterpretercoreInferShapeContext::GetInputVarPtrs(
    const std::string& name) const {
  const std::vector<Variable*>& vars = InputVars(name);
  paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize> res;
  res.reserve(vars.size());
  res.insert(res.begin(), vars.begin(), vars.end());
  return res;
}

paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize>
InterpretercoreInferShapeContext::GetOutputVarPtrs(
    const std::string& name) const {
  const std::vector<Variable*>& vars = OutputVars(name);
  paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize> res;
  res.reserve(vars.size());
  res.insert(res.begin(), vars.begin(), vars.end());
  return res;
}

DDim InterpretercoreInferShapeContext::GetInputDim(
    const std::string& name) const {
  const std::vector<Variable*>& vars = InputVars(name);
  PADDLE_ENFORCE_EQ(
      vars.size(),
      1UL,
      platform::errors::InvalidArgument(
          "Input(%s) should hold one element, but now it holds %zu elements.",
          name,
          vars.size()));
  return this->GetDim(vars[0]);
}

std::vector<DDim> InterpretercoreInferShapeContext::GetInputsDim(
    const std::string& name) const {
  const std::vector<Variable*>& vars = InputVars(name);
  return GetDims(vars);
}

proto::VarType::Type InterpretercoreInferShapeContext::GetInputVarType(
    const std::string& name) const {
  return GetVarType(InputVars(name).at(0));
}

std::vector<proto::VarType::Type>
InterpretercoreInferShapeContext::GetInputsVarType(
    const std::string& name) const {
  return GetVarTypes(InputVars(name));
}

std::vector<proto::VarType::Type>
InterpretercoreInferShapeContext::GetOutputsVarType(
    const std::string& name) const {
  return GetVarTypes(OutputVars(name));
}

void InterpretercoreInferShapeContext::SetOutputDim(const std::string& name,
                                                    const DDim& dim) {
  auto& vars = OutputVars(name);
  PADDLE_ENFORCE_EQ(
      vars.size(),
      1UL,
      platform::errors::InvalidArgument("Output(%s) should hold one element, "
                                        "but now it holds %zu elements.",
                                        name,
                                        vars.size()));
  SetDim(vars[0], dim);
}

void InterpretercoreInferShapeContext::SetOutputsDim(
    const std::string& name, const std::vector<DDim>& dims) {
  auto& vars = OutputVars(name);
  SetDims(vars, dims);
}

const phi::ArgumentMappingFn*
InterpretercoreInferShapeContext::GetPhiArgumentMappingFn() const {
  return phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_.Type());
}

const phi::KernelSignature*
InterpretercoreInferShapeContext::GetPhiDefaultKernelSignature() const {
  return &phi::DefaultKernelSignatureMap::Instance().Get(op_.Type());
}

void InterpretercoreInferShapeContext::SetSkipLoD(bool skip) {
  can_skip_lod_ = skip;
}

DDim InterpretercoreInferShapeContext::GetDim(Variable* var) const {
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::InvalidArgument("Input variable is nullptr."));
  if (var->IsType<LoDTensor>()) {
    return var->Get<LoDTensor>().dims();
  } else if (var->IsType<phi::SelectedRows>()) {
    return var->Get<phi::SelectedRows>().GetCompleteDims();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Only LoDTensor or SelectedRows support 'GetDim', but input "
        "Variable's type is %s.",
        ToTypeName(var->Type())));
  }
}

std::vector<DDim> InterpretercoreInferShapeContext::GetDims(
    const std::vector<Variable*>& vars) const {
  std::vector<DDim> ret;
  ret.reserve(vars.size());
  std::transform(
      vars.begin(), vars.end(), std::back_inserter(ret), [this](Variable* var) {
        return this->GetDim(var);
      });
  return ret;
}

std::vector<DDim> InterpretercoreInferShapeContext::GetRepeatedDims(
    const std::string& name) const {
  PADDLE_THROW(platform::errors::PreconditionNotMet(
      "GetRepeatedDims method only ban be used in compile time."));
}

void InterpretercoreInferShapeContext::SetDim(Variable* var, const DDim& dim) {
  if (var->IsType<LoDTensor>()) {
    var->GetMutable<LoDTensor>()->Resize(dim);
  } else if (var->IsType<phi::SelectedRows>()) {
    var->GetMutable<phi::SelectedRows>()->set_height(dim[0]);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Variable type error, expect LoDTensor or SelectedRows, but received "
        "(%s).",
        ToTypeName(var->Type())));
  }
}

void InterpretercoreInferShapeContext::SetDims(
    const std::vector<Variable*>& vars, const std::vector<DDim>& dims) {
  size_t length = vars.size();
  PADDLE_ENFORCE_EQ(length,
                    dims.size(),
                    platform::errors::InvalidArgument(
                        "The number of input variables do not match the "
                        "number of input dimensions, the number of variables "
                        "is %zu, the number of dimensions is %zu.",
                        length,
                        dims.size()));
  for (size_t i = 0; i < length; ++i) {
    if (vars[i] == nullptr) {
      continue;
    }
    SetDim(vars[i], dims[i]);
  }
}

void InterpretercoreInferShapeContext::SetRepeatedDims(
    const std::string& name, const std::vector<DDim>& dims) {
  PADDLE_THROW(platform::errors::PreconditionNotMet(
      "SetRepeatedDims method only can be used in compile time."));
}

std::vector<proto::VarType::Type> InterpretercoreInferShapeContext::GetVarTypes(
    const std::vector<Variable*>& vars) const {
  std::vector<proto::VarType::Type> retv;
  retv.resize(vars.size());
  std::transform(
      vars.begin(),
      vars.end(),
      retv.begin(),
      std::bind(std::mem_fn(&InterpretercoreInferShapeContext::GetVarType),
                this,
                std::placeholders::_1));
  return retv;
}

proto::VarType::Type InterpretercoreInferShapeContext::GetVarType(
    Variable* var) const {
  return ToVarType(var->Type());
}

const std::vector<Variable*>& InterpretercoreInferShapeContext::InputVars(
    const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  PADDLE_ENFORCE_NE(
      it,
      ctx_.inputs.end(),
      platform::errors::NotFound(
          "Operator (%s) does not have the input (%s).", op_.Type(), name));
  return it->second;
}

const std::vector<Variable*>& InterpretercoreInferShapeContext::OutputVars(
    const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  PADDLE_ENFORCE_NE(
      it,
      ctx_.outputs.end(),
      platform::errors::NotFound(
          "Operator (%s) does not have the outputs (%s).", op_.Type(), name));
  return it->second;
}

VariableScope::VariableScope(Scope* scope) {
  // for @EMPTY@ variable
  name2id_[kEmptyVarName] = kEmptyVarIndex;
  var_list_.push_back(nullptr);
  vec_meta_info_.emplace_back(0, nullptr);
  scope_ = scope;
  PADDLE_ENFORCE_NE(
      scope,
      nullptr,
      platform::errors::PreconditionNotMet(
          "You have passed a nullptr to construct VariableScope."));
}

VariableScope::~VariableScope() {}

Scope* VariableScope::GetMutableScope() const { return scope_; }

Scope* VariableScope::GetMutableLocalScope() const { return local_scope_; }

void VariableScope::SetScope(Scope* scope) { scope_ = scope; }

void VariableScope::SetLocalScope(Scope* local_scope) {
  VLOG(4) << "Set local scope: " << local_scope;
  local_scope_ = local_scope;
}

// Get variable id by name, return -1 if not found
int VariableScope::GetIdByName(const std::string& name) const {
  auto it = name2id_.find(name);
  if (it != name2id_.end()) {
    return it->second;
  }
  return -1;
}

// Get variable name by id, return "" if not found
std::string VariableScope::GetNameById(int id) const {
  // NOTE(zhiqiu): do not use vec_meta_info_[id].vardesc_->Name() since
  // vec_meta_info_[id] may be nullptr,
  // typically when the target variable is not existed in the original program
  // desc, but created by interpretercore.
  // For example, created and used by d2h_copy or h2d_copy operator.
  auto it = std::find_if(name2id_.begin(),
                         name2id_.end(),
                         [id](const auto& pair) { return pair.second == id; });
  if (it != name2id_.end()) {
    return it->first;
  }
  return "";
}

bool VariableScope::HasVar(const std::string& name) const {
  return name2id_.find(name) != name2id_.end();
}

int VariableScope::VarId(const std::string& name) const {
  CheckExist(name);
  return name2id_.at(name);
}

Variable* VariableScope::VarRef(int id) const { return var_list_[id]; }

size_t VariableScope::VarSize() const { return name2id_.size(); }

void VariableScope::AddVar(const std::string& name,
                           framework::VarDesc* var_desc) {
  if (!HasVar(name)) {
    auto id = VarSize();
    name2id_[name] = id;
    vec_meta_info_.emplace_back(0, var_desc);
    if (local_scope_ != nullptr) {
      var_list_.push_back(local_scope_->FindVar(name));
    } else {
      var_list_.push_back(scope_->FindVar(name));
    }
    PADDLE_ENFORCE_EQ(
        var_list_.size(),
        name2id_.size(),
        platform::errors::InvalidArgument(
            "The size of var_list and name2id map should be equal"));
  }
}

void VariableScope::SetVarDesc(const std::string& name,
                               framework::VarDesc* var_desc) {
  CheckExist(name);
  vec_meta_info_[VarId(name)].var_desc_ = var_desc;
}

paddle::framework::VarDesc* VariableScope::VarDesc(
    const std::string& name) const {
  return VarDesc(VarId(name));
}

paddle::framework::VarDesc* VariableScope::VarDesc(int id) const {
  CheckExist(id);
  return vec_meta_info_[id].var_desc_;
}

void VariableScope::SetVarSikpInplace(const std::string& name, bool skip) {
  CheckExist(name);
  vec_meta_info_[VarId(name)].sikp_inplace_ = skip;
}

bool VariableScope::GetVarSikpInplace(int id) const {
  CheckExist(id);
  return vec_meta_info_[id].sikp_inplace_;
}

void VariableScope::CheckExist(int id) const {
  PADDLE_ENFORCE_LT(id,
                    name2id_.size(),
                    platform::errors::PreconditionNotMet(
                        "Required var_id < %d, but received var_id = %d.",
                        name2id_.size(),
                        id));
}

void VariableScope::CheckExist(const std::string& name) const {
  PADDLE_ENFORCE_EQ(
      HasVar(name),
      true,
      platform::errors::NotFound("%s not in VariableScope.", name));
}

Instruction::Instruction(size_t id,
                         OpFuncNode&& op_func_node,
                         const platform::DeviceContext& dev_ctx)
    : id_(id), op_func_node_(op_func_node), dev_ctx_(dev_ctx) {
  PADDLE_ENFORCE_GE(id,
                    0,
                    platform::errors::PreconditionNotMet(
                        "Required id >= 0, but received id = %d", id));
}

size_t Instruction::Id() const { return id_; }

const std::map<std::string, std::vector<int>>& Instruction::Inputs() const {
  return op_func_node_.input_index;
}

const std::map<std::string, std::vector<int>>& Instruction::Outputs() const {
  return op_func_node_.output_index;
}

const std::unordered_set<int>& Instruction::NoDataTransformVars() const {
  return op_func_node_.no_data_transform_index;
}

OpKernelComputeFunc Instruction::KernelFunc() const {
  return op_func_node_.kernel_func_;
}

phi::Kernel* Instruction::PhiKernel() const {
  return op_func_node_.phi_kernel_;
}

OpFuncType Instruction::KernelType() const { return op_func_node_.type_; }

const std::map<int, int>& Instruction::InplaceBackMap() const {
  return op_func_node_.inplace_back_map;
}

OperatorBase* Instruction::OpBase() const {
  auto op_base = op_func_node_.operator_base_;
  PADDLE_ENFORCE_NOT_NULL(
      op_base,
      platform::errors::PreconditionNotMet("op_base shall not be nullptr."));
  return op_base.get();
}

NextInstruction& Instruction::NextInstructions() { return next_instruction_; }

const NextInstruction& Instruction::NextInstructions() const {
  return next_instruction_;
}

void Instruction::AddGCCheckVar(size_t id) { gc_check_var_list_.push_back(id); }

const std::vector<size_t>& Instruction::GCCheckVars() const {
  return gc_check_var_list_;
}

void Instruction::ResetContext(const VariableValueMap& in_vars,
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

void Instruction::ResetContextWithScope(const VariableValueMap& in_vars,
                                        const VariableValueMap& out_vars,
                                        const framework::Scope& scope) {
  runtime_ctx_.reset(new RuntimeContext(in_vars, out_vars));
  infershape_ctx_.reset(
      new InterpretercoreInferShapeContext(*OpBase(), *runtime_ctx_.get()));
  execution_ctx_.reset(
      new ExecutionContext(*OpBase(), scope, dev_ctx_, *runtime_ctx_.get()));
}

std::shared_ptr<RuntimeContext> Instruction::InnerRuntimeContext() const {
  return runtime_ctx_;
}

std::shared_ptr<InterpretercoreInferShapeContext>
Instruction::InnerInferShapeContext() const {
  return infershape_ctx_;
}

std::shared_ptr<ExecutionContext> Instruction::InnerExecutionContext() const {
  return execution_ctx_;
}

const platform::DeviceContext& Instruction::DeviceContext() const {
  return dev_ctx_;
}

const std::vector<std::pair<Variable*, Variable*>>& Instruction::InplaceInfo()
    const {
  return vec_inplace_in_to_out_;
}

void Instruction::AddInplace(Variable* in, Variable* out) {
  vec_inplace_in_to_out_.emplace_back(in, out);
}

void Instruction::ClearInplace() { vec_inplace_in_to_out_.clear(); }

const std::vector<EventInter>& Instruction::InputEvents() const {
  return intput_events_;
}

const std::vector<EventInter>& Instruction::OutputEvents() const {
  return output_events_;
}

void Instruction::AddInputEvent(size_t var_id,
                                std::shared_ptr<platform::DeviceEvent> event,
                                platform::DeviceType waiter_type) {
  intput_events_.emplace_back(var_id, event, waiter_type);
}

void Instruction::AddOutputEvent(size_t var_id,
                                 std::shared_ptr<platform::DeviceEvent> event,
                                 platform::DeviceType waiter_type) {
  output_events_.emplace_back(var_id, event, waiter_type);
}

}  // namespace framework
}  // namespace paddle
