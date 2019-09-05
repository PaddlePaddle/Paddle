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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(benchmark);
DECLARE_bool(check_nan_inf);
DEFINE_int32(inner_op_parallelism, 0, "number of threads for inner op");
DEFINE_bool(fast_check_nan_inf, false,
            "Fast checking NAN/INF after each operation. It will be a little"
            "bit slow, much faster than check_nan_inf");

namespace paddle {
namespace framework {

std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority = {
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kCUDNN),
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kPlain),
    std::make_tuple(platform::CPUPlace(), LibraryType::kMKLDNN),
    std::make_tuple(platform::CPUPlace(), LibraryType::kPlain),
};

proto::VarType::Type GetDataTypeOfVar(const Variable* var) {
  if (var->IsType<framework::LoDTensor>()) {
    return var->Get<framework::LoDTensor>().type();
  } else if (var->IsType<framework::SelectedRows>()) {
    return var->Get<framework::SelectedRows>().value().type();
  } else {
    PADDLE_THROW("Var should be LoDTensor or SelectedRows");
  }
}

static DDim GetDimsDebug(const Scope& scope, const std::string& name,
                         bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    return tensor.dims();
  } else if (var->IsType<SelectedRows>()) {
    if (get_actual_dim) {
      return var->Get<SelectedRows>().value().dims();
    } else {
      return var->Get<SelectedRows>().GetCompleteDims();
    }
  } else {
    return DDim({-1});
  }
}

static bool VarInited(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) return false;
  return var->IsInitialized();
}

static std::string GetDtype(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return "";
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "";
    }
    return DataTypeToString(tensor.type());
  } else if (var->IsType<SelectedRows>()) {
    auto tensor = var->Get<SelectedRows>().value();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return "uninited";
    } else {
      return DataTypeToString(tensor.type());
    }
  } else {
    return "";
  }
}

static int GetRowSize(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return -1;
  }

  if (var->IsType<SelectedRows>()) {
    return var->Get<SelectedRows>().rows().size();
  }

  return -1;
}

static LoD GetLoDDebug(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<LoDTensor>()) {
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (UNLIKELY(!tensor.IsInitialized())) {
      return default_lod;
    }
    return tensor.lod();
  } else {
    return default_lod;
  }
}

RuntimeContext::RuntimeContext(const VariableNameMap& innames,
                               const VariableNameMap& outnames,
                               const Scope& scope) {
  for (auto& var_name_item : innames) {
    std::vector<Variable*>& input_vars = inputs[var_name_item.first];
    input_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      input_vars.push_back(scope.FindVar(var_name));
    }
  }
  for (auto& var_name_item : outnames) {
    std::vector<Variable*>& output_vars = outputs[var_name_item.first];
    output_vars.reserve(var_name_item.second.size());
    for (auto& var_name : var_name_item.second) {
      output_vars.push_back(scope.FindVar(var_name));
    }
  }
}

void OperatorBase::Run(const Scope& scope, const platform::Place& place) {
  try {
    VLOG(4) << place << " " << DebugStringEx(&scope);
    if (platform::is_gpu_place(place)) {
#ifndef PADDLE_WITH_CUDA
      PADDLE_THROW("Cannot run operator on place %s", place);
#else
      auto dev_id = boost::get<platform::CUDAPlace>(place).device;
      platform::SetDeviceId(dev_id);
#endif
    }

    // The profile has a process-wide mutex, results in serious performance
    // issue
    // in concurrency scenerio. Here use an `if` to fix this issue.
    // Please not remove the `if`, ask @Superjomn if there are any concern.
    if (platform::IsProfileEnabled()) {
      platform::RecordEvent record_event(Type());
      RunImpl(scope, place);
    } else {
      RunImpl(scope, place);
    }
    VLOG(3) << place << " " << DebugStringEx(&scope);
  } catch (platform::EnforceNotMet exception) {
    framework::InsertCallStackInfo(Type(), Attrs(), &exception);
    throw std::move(exception);
  } catch (...) {
    std::rethrow_exception(std::current_exception());
  }
}

bool OperatorBase::HasInputs(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(ins.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    type_, name);
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Operator %s does not have the input %s.",
                 type_, name);
  return it->second;
}

bool OperatorBase::HasOutputs(const std::string& name) const {
  if (outputs_.find(name) != outputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(outs.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    type_, name);
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(),
                 "Operator %s does not have an output called %s.", type_, name);
  return it->second;
}

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      auto var_name = input.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, var_name);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = GetDtype(*scope, var_name);
          ss << ":" << dtype;
          ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
          ss << "(" << GetLoDDebug(*scope, var_name) << ")";
        }
      }
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      auto var_name = output.second[i];
      ss << var_name;
      if (scope) {
        if (!VarInited(*scope, var_name)) {
          ss << "[uninited]";
        } else {
          int row_size = GetRowSize(*scope, output.second[i]);
          if (row_size >= 0) {
            ss << "[row_size=" << row_size << "]";
          }
          std::string dtype = GetDtype(*scope, output.second[i]);
          ss << ":" << dtype;
          ss << "[" << GetDimsDebug(*scope, var_name, true) << "]";
          ss << "(" << GetLoDDebug(*scope, var_name) << ")";
        }
      }
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type),
      inputs_(inputs),
      outputs_(outputs),
      attrs_(attrs),
      // NOTE(zjl): why op_info may be nullptr?
      info_(OpInfoMap::Instance().GetNullable(type)) {
  GenerateTemporaryNames();
  CheckAllInputOutputSet();
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto& info = Info();

  // get all OpProto::Var for outputs
  for (auto& o : info.Proto().outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

void OperatorBase::CheckAllInputOutputSet() const {
  if (info_ == nullptr || info_->proto_ == nullptr) return;

  for (auto& in : info_->Proto().inputs()) {
    if (!in.dispensable()) {
      PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
                     "Operator %s's input, %s, is not set", Type(), in.name());
    }
  }

  for (auto& out : info_->Proto().outputs()) {
    if (!out.dispensable()) {
      PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
                     "Operator %s's output, %s, is not set", Type(),
                     out.name());
    }
  }
}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

static bool VarIsTensor(const Variable& var) {
  return var.IsType<LoDTensor>() || var.IsType<SelectedRows>();
}

const Tensor* GetLoDTensorOrSelectedRowsValueFromVar(const Variable& var) {
  if (var.IsType<LoDTensor>()) {
    return static_cast<const Tensor*>(&(var.Get<LoDTensor>()));
  } else if (var.IsType<SelectedRows>()) {
    return &(var.Get<SelectedRows>().value());
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 ToTypeName(var.Type()));
  }
}

Tensor* GetMutableLoDTensorOrSelectedRowsValueFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 ToTypeName(var->Type()));
  }
}

bool ExecutionContext::HasInput(const std::string& name) const {
  if (!op_.HasInputs(name)) {
    return false;
  }
  auto& ins = Inputs(name);
  size_t length = ins.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Input %s should not have more than one inputs", name);
  auto arg = ins[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

bool ExecutionContext::HasOutput(const std::string& name) const {
  if (!op_.HasOutputs(name)) {
    return false;
  }
  auto& outs = Outputs(name);
  size_t length = outs.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Output %s should not have more than one inputs", name);
  auto arg = outs[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

const Variable* ExecutionContext::InputVar(const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(it->second.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    op_.Type(), name);
  return it->second.empty() ? nullptr : it->second[0];
}

Variable* ExecutionContext::OutputVar(const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  if (it == ctx_.outputs.end()) return nullptr;

  PADDLE_ENFORCE_LE(it->second.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    op_.Type(), name);
  return it->second.empty() ? nullptr : it->second[0];
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  return Input<LoDTensor>(name);
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  auto it = ctx_.inputs.find(name);
  if (it == ctx_.inputs.end()) {
    return {};
  }
  const std::vector<Variable*>& vars = it->second;
  std::vector<const Tensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                 [&](Variable* var) -> const Tensor* {
                   if (var == nullptr) return nullptr;
                   PADDLE_ENFORCE(
                       var->IsType<LoDTensor>(),
                       "should be LoDTensor, but the received type is %s",
                       ToTypeName(var->Type()));
                   return &(var->Get<LoDTensor>());
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  return Output<LoDTensor>(name);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto it = ctx_.outputs.find(name);
  if (it == ctx_.outputs.end()) {
    return {};
  }
  const std::vector<Variable*>& vars = it->second;
  std::vector<Tensor*> res;
  res.reserve(vars.size());
  std::transform(vars.begin(), vars.end(), std::back_inserter(res),
                 [&](Variable* var) -> Tensor* {
                   return var == nullptr ? nullptr
                                         : var->GetMutable<LoDTensor>();
                 });
  return res;
}

bool OpSupportGPU(const std::string& op_type) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it == all_kernels.end()) {
    // All control operator must support GPU
    return true;
  }
  for (auto& kern_pair : it->second) {
    if (platform::is_gpu_place(kern_pair.first.place_)) {
      return true;
    }
  }
  return false;
}

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const Scope& scope,
                           const RuntimeContext& ctx)
      : op_(op), ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    // has only one input
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end()) {
      return false;
    }
    const auto& in = it->second;
    if (in.size() == 0) return false;
    PADDLE_ENFORCE_EQ(in.size(), 1UL,
                      "Input %s should not have more than one inputs", name);
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
    PADDLE_ENFORCE_EQ(out.size(), 1UL,
                      "Output %s should not have more than one outputs", name);
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

  const std::vector<std::string>& Inputs(
      const std::string& name) const override {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(
      const std::string& name) const override {
    return op_.Outputs(name);
  }

  void ShareDim(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE(in_it != ctx_.inputs.end() && in_it->second.size() > i,
                   "Inputs %s should have %llu argument", in, i);
    PADDLE_ENFORCE(out_it != ctx_.outputs.end() && out_it->second.size() > j,
                   "Outputs %s should have %llu argument", out, j);

    Variable* in_var = in_it->second[i];
    Variable* out_var = out_it->second[j];

    PADDLE_ENFORCE(in_var->Type() == out_var->Type(),
                   "The type of %s and %s is not the same.", in, out);

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
      PADDLE_THROW(
          "Currently, the input type of ShareDim only can be LoDTensor "
          "or SelectedRows.");
    }
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE(in_it != ctx_.inputs.end() && in_it->second.size() > i,
                   "Inputs %s should have %llu argument", in, i);
    PADDLE_ENFORCE(out_it != ctx_.outputs.end() && out_it->second.size() > j,
                   "Outputs %s should have %llu argument", out, j);

    Variable* in_var = in_it->second.at(i);
    if (!in_var->IsType<LoDTensor>()) return;
    Variable* out_var = out_it->second.at(j);
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
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

  void DecreaseLoDLevel(const std::string& in, const std::string& out,
                        size_t i = 0, size_t j = 0) const override {
    PADDLE_THROW("DecreaseLoDLevel is only used in compile time.");
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
    PADDLE_ENFORCE_EQ(vars.size(), 1UL,
                      "Input(%s) should hold one element, but now it holds %d",
                      name, vars.size());
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
    PADDLE_ENFORCE_EQ(vars.size(), 1UL,
                      "Output(%s) should hold one element, but now it holds %d",
                      name, vars.size());
    SetDim(vars[0], dim);
  }

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override {
    auto& vars = OutputVars(name);
    SetDims(vars, dims);
  }

 protected:
  DDim GetDim(Variable* var) const {
    PADDLE_ENFORCE_NOT_NULL(var);
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(
          "Only LoDTensor/SelectedRows support 'GetDim', but Variables "
          "type_id is %s.",
          ToTypeName(var->Type()));
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
    PADDLE_THROW("Only compile time support this method");
  }

  void SetDim(Variable* var, const DDim& dim) {
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                   ToTypeName(var->Type()));
    }
  }

  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims) {
    size_t length = vars.size();
    PADDLE_ENFORCE_EQ(length, dims.size());
    for (size_t i = 0; i < length; ++i) {
      if (vars[i] == nullptr) {
        continue;
      }
      SetDim(vars[i], dims[i]);
    }
  }

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW("Only compile time support this method");
  }

  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<Variable*>& vars) const {
    std::vector<proto::VarType::Type> retv;
    retv.resize(vars.size());
    std::transform(vars.begin(), vars.end(), retv.begin(),
                   std::bind(std::mem_fn(&RuntimeInferShapeContext::GetVarType),
                             this, std::placeholders::_1));
    return retv;
  }

  proto::VarType::Type GetVarType(Variable* var) const {
    return ToVarType(var->Type());
  }

 private:
  const std::vector<Variable*>& InputVars(const std::string& name) const {
    auto it = ctx_.inputs.find(name);
    PADDLE_ENFORCE(it != ctx_.inputs.end(),
                   "Operator %s does not have the input %s.", op_.Type(), name);
    return it->second;
  }

  const std::vector<Variable*>& OutputVars(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    PADDLE_ENFORCE(it != ctx_.outputs.end(),
                   "Operator %s does not have the outputs %s.", op_.Type(),
                   name);
    return it->second;
  }

  const OperatorBase& op_;
  const RuntimeContext& ctx_;
};

static void CheckTensorNANOrInf(const std::string& op_type,
                                const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type() != proto::VarType::FP32 &&
      tensor.type() != proto::VarType::FP64) {
    return;
  }
  PADDLE_ENFORCE(!framework::TensorContainsInf(tensor),
                 "Operator %s output Tensor %s contains Inf", op_type, name);
  PADDLE_ENFORCE(!framework::TensorContainsNAN(tensor),
                 "Operator %s output Tensor %s contains NAN", op_type, name);
}

void OperatorWithKernel::RuntimeInferShape(const Scope& scope,
                                           const platform::Place& place,
                                           const RuntimeContext& ctx) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, scope, ctx);
  this->InferShape(&infer_shape_ctx);
}

std::vector<KernelConfig>* OperatorWithKernel::GetKernelConfig(
    const OpKernelType& key) const {
  auto config_iter = kernel_configs_map_.find(key);
  std::vector<KernelConfig>* kernel_configs = nullptr;
  if (config_iter != kernel_configs_map_.end()) {
    kernel_configs = &(config_iter->second);
  }
  return kernel_configs;
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place) const {
  // To reduce the elapsed time of HasAttr, we use bool variable to record the
  // result of HasAttr.
  if (!enable_cache_runtime_context_ && HasAttr(kEnableCacheRuntimeContext))
    enable_cache_runtime_context_ = true;
  if (!all_kernels_must_compute_runtime_shape_ &&
      HasAttr(kAllKernelsMustComputeRuntimeShape))
    all_kernels_must_compute_runtime_shape_ = true;
  if (!enable_cache_runtime_context_) {
    RuntimeContext ctx(Inputs(), Outputs(), scope);
    RunImpl(scope, place, &ctx);
  } else {
    const Scope* cur_scope = &scope;
    if (runtime_ctx_.get() == nullptr || pre_scope_ != cur_scope) {
      std::lock_guard<std::mutex> lock(cache_update_mutex_);
      if (runtime_ctx_.get() == nullptr || pre_scope_ != cur_scope) {
        runtime_ctx_.reset(new RuntimeContext(Inputs(), Outputs(), scope));
        pre_scope_ = cur_scope;
      }
    }
    RunImpl(scope, place, runtime_ctx_.get());
  }
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place,
                                 RuntimeContext* runtime_ctx) const {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  if (kernel_type_.get() == nullptr || kernel_func_.get() == nullptr) {
    ChooseKernel(*runtime_ctx, scope, place);
  }

  std::vector<KernelConfig>* kernel_configs = GetKernelConfig(*kernel_type_);

  // do data transformScope &transfer_scope;
  std::vector<std::string> transfered_inplace_vars;
  auto* transfer_scope =
      PrepareData(scope, *kernel_type_, &transfered_inplace_vars, runtime_ctx);

  // exec scope is the scope that kernel actually executed on.
  const Scope& exec_scope =
      (transfer_scope == nullptr ? scope : *transfer_scope);

  if (!(kernel_type_->place_ == dev_ctx->GetPlace())) {
    dev_ctx = pool.Get(kernel_type_->place_);
  }

  if (!all_kernels_must_compute_runtime_shape_) {
    RuntimeInferShapeContext infer_shape_ctx(*this, exec_scope, *runtime_ctx);
    this->InferShape(&infer_shape_ctx);
  }
  // TODO(panyx0718): ExecutionContext should only depend on RuntimeContext
  // not Scope. Imperative mode only pass inputs and get outputs.
  (*kernel_func_)(ExecutionContext(*this, exec_scope, *dev_ctx, *runtime_ctx,
                                   kernel_configs));

  if (!transfered_inplace_vars.empty()) {
    // there is inplace variable has been transfered.
    TransferInplaceVarsBack(scope, transfered_inplace_vars, *transfer_scope);
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    dev_ctx->Wait();
  }

  if (FLAGS_fast_check_nan_inf) {
    for (auto& vname : OutputVars(true)) {
      // only check inserted vars,
      // please see executor.py for details of fast_check_nan_inf
      if (vname.rfind("debug_var") == 0) {
        VLOG(3) << "debugging nan/inf in var " << vname;

        auto* var = exec_scope.FindVar(vname);
        if (var == nullptr) continue;
        if (var->IsType<framework::LoDTensor>()) {
          CheckTensorNANOrInf(type_, vname, var->Get<framework::LoDTensor>());
        } else if (var->IsType<framework::SelectedRows>()) {
          CheckTensorNANOrInf(type_, vname,
                              var->Get<framework::SelectedRows>().value());
        }
      }
    }
  }

  if (FLAGS_check_nan_inf) {
    for (auto& vname : OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      if (var->IsType<framework::LoDTensor>()) {
        CheckTensorNANOrInf(type_, vname, var->Get<framework::LoDTensor>());
      } else if (var->IsType<framework::SelectedRows>()) {
        CheckTensorNANOrInf(type_, vname,
                            var->Get<framework::SelectedRows>().value());
      }
    }
  }

  // To solve issue #15032, have a discussion with @Luotao for cpu inference,
  // do not cache transfer scope, hence in this case delete transfer scope
  // after run to avoid memory leak
  if (transfer_scope && !run_by_executor_ && !enable_cache_transfer_scope_) {
    scope.DeleteScope(transfer_scope);
  }
}

void OperatorWithKernel::ChooseKernel(const RuntimeContext& ctx,
                                      const Scope& scope,
                                      const platform::Place& place) const {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  OpKernelMap& kernels = kernels_iter->second;

  auto expected_kernel_key = this->GetExpectedKernelType(
      ExecutionContext(*this, scope, *dev_ctx, ctx, nullptr));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
  // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
  if (kernel_iter == kernels.end() &&
      expected_kernel_key.library_type_ == LibraryType::kMKLDNN) {
    VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
    expected_kernel_key.library_type_ = LibraryType::kPlain;
    expected_kernel_key.data_layout_ = DataLayout::kAnyLayout;
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("op %s does not have kernel for %s", type_,
                 KernelTypeToString(expected_kernel_key));
  }

  std::lock_guard<std::mutex> lock(cache_update_mutex_);
  if (kernel_type_.get() == nullptr || kernel_func_.get() == nullptr) {
    kernel_type_.reset(new OpKernelType(expected_kernel_key));
    kernel_func_.reset(new OpKernelFunc(kernel_iter->second));
  }
}

void OperatorWithKernel::TransferInplaceVarsBack(
    const Scope& scope, const std::vector<std::string>& inplace_vars,
    const Scope& transfer_scope) const {
  for (auto& var_name : inplace_vars) {
    VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
    auto* origin_var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(origin_var, "The var[%s] should not be nullptr.",
                            var_name);
    auto* original_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(origin_var);
    auto* var = transfer_scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(var, "The var[%s] should not be nullptr.",
                            var_name);
    auto* transformed_tensor = GetLoDTensorOrSelectedRowsValueFromVar(*var);
    original_tensor->ShareDataWith(*transformed_tensor);
  }
}

Scope* OperatorWithKernel::PrepareData(
    const Scope& scope, const OpKernelType& expected_kernel_key,
    std::vector<std::string>* transfered_inplace_vars,
    RuntimeContext* ctx) const {
  Scope* new_scope = nullptr;

  std::unordered_set<std::string> no_buffer_ins;
  if (info_) {
    auto& no_buffer_inferer = info_->NoNeedBufferVarsInferer();
    // Some op may not register NoNeedBufferVarsInferer
    if (no_buffer_inferer) {
      no_buffer_ins = no_buffer_inferer(Inputs(), Outputs(), Attrs());
    }
  }

  for (auto& var_name_item : Inputs()) {
    // NOTE(zjl): STL does not guarantee fast std::unordered_set::count when set
    // is empty. At least STL implemented on my mac does calculate hash code
    // of search key even though the set is empty.
    if (!no_buffer_ins.empty() &&
        no_buffer_ins.count(var_name_item.first) > 0) {
      VLOG(7) << "Skip scanning input " << var_name_item.first
              << " in Operator " << type_;
      continue;
    }

    std::vector<Variable*>& input_vars = ctx->inputs[var_name_item.first];

    for (size_t i = 0; i < var_name_item.second.size(); ++i) {
      auto& var_name = var_name_item.second[i];
      auto* var = input_vars[i];

      // Only tensor can be tranfer to another device.
      if (var == nullptr || !VarIsTensor(*var)) {
        continue;
      }

      auto* tensor_in = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      if (!tensor_in->IsInitialized()) {
        continue;
      }

      auto kernel_type_for_var = GetKernelTypeForVar(
          var_name_item.first, *tensor_in, expected_kernel_key);

      if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
        continue;
      }

      auto out_var_names = OutputVars(true);
      if (std::find(out_var_names.begin(), out_var_names.end(), var_name) !=
          out_var_names.end()) {
        transfered_inplace_vars->emplace_back(var_name);
      }

      VLOG(3) << "Transform Variable " << var_name << " from "
              << kernel_type_for_var << " to " << expected_kernel_key;

      // In the inference scenerio, the scopes will be reused across the
      // batches, so the `new_scope` here will result in GPU memroy explosion
      // over the  running of operators.
      // We use a thread_local cache to fix that issue, the key in the cache is
      // the combination of the `scope` argument, from_kernel_type,
      // target_kernel_type.
      // Have a discussion with @Superjomn or the inference developers if some
      // changes on this logic for this macro might not tested on the other
      // scenerios.
      // If this op is not called by an Executor or ParallelExecutor, it should
      // called by a NaiveExecutor, the NaiveExecutor will cache the scopes and
      // variables, that behavior a lot different.
      //
      // To solve issue #15032, have a discussion with @Luotao for cpu
      // inference, for all cpu kernels cases without GPU participation, here
      // not do transfer scope caching, and cpu inference performance is not
      // impacted by test.
      enable_cache_transfer_scope_ = false;
      if (!run_by_executor_ &&
          (platform::is_gpu_place(kernel_type_for_var.place_) ||
           platform::is_gpu_place(expected_kernel_key.place_))) {
        new_scope = TryCreateTransferScope(kernel_type_for_var,
                                           expected_kernel_key, &scope);
        enable_cache_transfer_scope_ = true;
      }
      if (!new_scope) {
        new_scope = &scope.NewScope();
      }
      // For inference, if a gpu model has an op which could only run on CPU,
      // each result of different input will be the same with the first one.
      // The reason is that if a gpu tensor is the input of a cpu kernel,
      // we will create a new cpu tensor in new scope.
      // However, if enable_cache_runtime_context_, we get the cpu tensor each
      // time, not the gpu tensor.
      // Thus, we set pre_scope_ = nullptr to trigger `new RuntimeContext()` in
      // RunImpl().
      if (enable_cache_runtime_context_) {
        pre_scope_ = nullptr;
      }

      auto* trans_var = new_scope->Var(var_name);
      input_vars[i] = trans_var;

      Tensor out;
      TransformData(expected_kernel_key, kernel_type_for_var, *tensor_in, &out);
      SetTensorToVariable(*var, out, trans_var);
    }
  }

  return new_scope;
}

proto::VarType::Type OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  proto::VarType::Type dafault_data_type =
      static_cast<proto::VarType::Type>(-1);
  proto::VarType::Type data_type = dafault_data_type;
  for (auto& input : this->inputs_) {
    const std::vector<const Variable*> vars = ctx.MultiInputVar(input.first);
    for (size_t i = 0; i < vars.size(); ++i) {
      const Variable* var = vars[i];
      if (var != nullptr) {
        const Tensor* t = nullptr;
        if (var->IsType<Tensor>()) {
          t = &var->Get<Tensor>();
        } else if (var->IsType<LoDTensor>()) {
          t = &var->Get<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
          t = &(var->Get<SelectedRows>().value());
        }
        if (t != nullptr) {
          PADDLE_ENFORCE(t->IsInitialized(), "Input %s(%lu) is not initialized",
                         input.first, i);
          proto::VarType::Type tmp = t->type();
          PADDLE_ENFORCE(
              tmp == data_type || data_type == dafault_data_type,
              "DataType of Paddle Op %s %s must be the same. Get (%s) != (%s)",
              Type(), input.first, DataTypeToString(data_type),
              DataTypeToString(tmp));
          data_type = tmp;
        }
      }
    }
  }
  PADDLE_ENFORCE(data_type != dafault_data_type,
                 "DataType should be indicated by input");
  return data_type;
}

OpKernelType OperatorWithKernel::GetExpectedKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.GetPlace());
}

OpKernelType OperatorWithKernel::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const OpKernelType& expected_kernel_type) const {
  return OpKernelType(expected_kernel_type.data_type_, tensor.place(),
                      tensor.layout());
}

}  // namespace framework
}  // namespace paddle
