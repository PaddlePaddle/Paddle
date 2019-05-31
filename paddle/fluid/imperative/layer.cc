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

#include "paddle/fluid/imperative/layer.h"
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/imperative/prepared_operator.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

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

static framework::VariableNameMap CreateVarNameMap(
    const framework::OpInfo* op_info, const std::string& op_type,
    const NameVarBaseMap& varbase_map, bool is_input) {
  if (op_info == nullptr || op_info->proto_ == nullptr) {
    return {};
  }

  VLOG(2) << "CreateVarNameMap " << is_input;
  framework::VariableNameMap result;

  for (auto& var :
       is_input ? op_info->Proto().inputs() : op_info->Proto().outputs()) {
    auto it = varbase_map.find(var.name());
    if (it == varbase_map.end()) {
      PADDLE_ENFORCE(var.dispensable());
      result[var.name()] = {};
    } else {
      auto& var_vector = it->second;
      std::vector<std::string> args;
      args.reserve(var_vector.size());
      for (auto& var_base : var_vector) {
        args.emplace_back(var_base->Name());
      }
      result[var.name()] = std::move(args);
    }
  }

  VLOG(2) << "CreateVarNameMap " << is_input << " done";
  return result;
}

static framework::RuntimeContext PrepareRuntimeContext(
    const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  framework::VariableValueMap inputs, outputs;
  for (auto& in_pair : ins) {
    auto& in_ctx = inputs[in_pair.first];
    in_ctx.reserve(in_pair.second.size());
    for (auto& in_var : in_pair.second) {
      in_ctx.emplace_back(in_var->MutableVar());
    }
  }

  for (auto& out_pair : outs) {
    auto& out_ctx = outputs[out_pair.first];
    out_ctx.reserve(out_pair.second.size());
    for (auto& out_var : out_pair.second) {
      out_ctx.emplace_back(out_var->MutableVar());
    }
  }
  return framework::RuntimeContext(std::move(inputs), std::move(outputs));
}

static std::string DebugString(
    const std::string& name,
    const std::vector<std::shared_ptr<VarBase>>& vars) {
  std::stringstream ss;
  ss << name << "{";

  for (size_t i = 0; i < vars.size(); ++i) {
    if (i > 0) ss << ", ";

    if (vars[i] == nullptr) {
      ss << "NULL";
      continue;
    }
    ss << vars[i]->Name() << "[";
    auto& var = vars[i]->Var();
    if (!var.IsInitialized()) {
      ss << "NOT_INITED_VAR";
    } else if (var.IsType<framework::LoDTensor>()) {
      auto& tensor = var.Get<framework::LoDTensor>();
      ss << "LoDTensor<";
      if (tensor.IsInitialized()) {
        ss << framework::DataTypeToString(tensor.type()) << ", ";
        ss << tensor.place() << ", ";
        ss << "(" << tensor.dims() << ")";
      } else {
        ss << "NOT_INITED";
      }
      ss << ">";
    } else {
      ss << "UNRESOLVED_TYPE";
    }
    ss << "]";
  }

  ss << "}";
  return ss.str();
}

static std::string DebugString(const std::string& op_type,
                               const NameVarBaseMap& ins,
                               const NameVarBaseMap& outs) {
  std::stringstream ss;
  ss << "Op(" << op_type << "): ";

  ss << "Inputs: ";

  size_t i = 0;
  for (auto& pair : ins) {
    if (i > 0) ss << ", ";
    ss << DebugString(pair.first, pair.second);
    ++i;
  }

  ss << ",   Outputs: ";
  i = 0;
  for (auto& pair : outs) {
    if (i > 0) ss << ", ";
    ss << DebugString(pair.first, pair.second);
    ++i;
  }
  return ss.str();
}

static std::vector<std::unique_ptr<framework::OpDesc>> CreateGradOpDescs(
    const framework::OpInfo& op_info, const framework::OpDesc& op_desc,
    const std::unordered_set<std::string>& no_grad_set,
    const std::vector<framework::BlockDesc*>& grad_sub_block,
    std::unordered_map<std::string, std::string>* grad_to_var) {
  if (op_info.grad_op_maker_) {
    return op_info.grad_op_maker_(op_desc, no_grad_set, grad_to_var,
                                  grad_sub_block);
  } else {
    return {};
  }
}

void VarBase::SetGeneratedOp(OpBase* op) {
  generated_op_ = op ? op->shared_from_this() : nullptr;
}

void VarBase::ClearGradient() {
  if (grad_var_) {
    auto* grad_t = grad_var_->var_.GetMutable<framework::LoDTensor>();
    if (grad_t->IsInitialized()) {
      auto* dev_ctx =
          platform::DeviceContextPool::Instance().Get(grad_t->place());
      operators::math::set_constant(*dev_ctx, grad_t, 0.0);
    }
  }
}

std::shared_ptr<VarBase> VarBase::NewVarBase(const platform::Place& dst_place,
                                             const bool blocking) const {
  PADDLE_ENFORCE(var_.IsInitialized() && var_.IsType<framework::LoDTensor>(),
                 "Variable must be initialized and type of LoDTensor when "
                 "getting numpy tensor");

  // TODO(minqiyang): change this after move unique_name generator to CXX
  auto& src_tensor = var_.Get<framework::LoDTensor>();

  auto new_var = std::make_shared<VarBase>(false);

  auto* dst_tensor = new_var->var_.GetMutable<framework::LoDTensor>();
  dst_tensor->set_lod(src_tensor.lod());

  framework::TensorCopy(src_tensor, dst_place, dst_tensor);
  if (blocking) {
    platform::DeviceContextPool::Instance().Get(dst_place)->Wait();
    auto src_place = src_tensor.place();
    if (!(src_place == dst_place)) {
      platform::DeviceContextPool::Instance().Get(src_place)->Wait();
    }
  }

  if (platform::is_gpu_place(dst_place)) {
    VLOG(3) << "copy tensor " << Name() << " from gpu";
  }

  return new_var;
}

OpBase::OpBase(Tracer* tracer, size_t id, const std::string& type,
               const NameVarBaseMap& ins, const NameVarBaseMap& outs,
               framework::AttributeMap attrs, const platform::Place& place)
    : holded_tracer_(tracer), id_(id), place_(place) {
  auto* info = &(framework::OpInfoMap::Instance().Get(type));

  // Step 1: Run forward
  if (info->Checker() != nullptr) {
    info->Checker()->Check(&attrs);
  }

  auto input_name_map = CreateVarNameMap(info, type, ins, true);
  auto output_name_map = CreateVarNameMap(info, type, outs, false);
  op_ = framework::OpRegistry::CreateOp(type, std::move(input_name_map),
                                        std::move(output_name_map),
                                        std::move(attrs));
}

OpBase::OpBase(Tracer* tracer, size_t id, const framework::OpDesc& op_desc,
               const NameVarBaseMap& ins, const NameVarBaseMap& outs,
               const platform::Place& place)
    : holded_tracer_(tracer),
      id_(id),
      op_(framework::OpRegistry::CreateOp(op_desc)),
      place_(place) {}

void OpBase::Run(const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  auto* op_kernel = dynamic_cast<framework::OperatorWithKernel*>(op_.get());
  PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");
  VLOG(2) << "Create op " << Type() << " with input " << ins.size()
          << " and output " << outs.size();
  auto& info = op_->Info();
  if (info.infer_var_type_) {
    RuntimeInferVarTypeContext infer_var_type_ctx(ins, &outs, op_->Attrs());
    info.infer_var_type_(&infer_var_type_ctx);
  }

  // Initialize output var type
  for (auto& var_pair : outs) {
    for (auto& var : var_pair.second) {
      InitializeVariable(var->MutableVar(), var->Type());
    }
  }

  VLOG(3) << "Running Op " << Type();
  VLOG(1) << DebugString(Type(), ins, outs);
  auto runtime_ctx = PrepareRuntimeContext(ins, outs);
  auto runtime_place = PreparedOp::GetExpectedPlace(place(), ins);

  auto prepared_op =
      PreparedOp::Prepare(runtime_ctx, *op_kernel, runtime_place);

  prepared_op.Run();

  VLOG(3) << "Running Op " << Type() << " ends";
  VLOG(1) << DebugString(Type(), ins, outs);
}

void OpBase::TraceBackward(const framework::OpDesc& fwd_op,
                           const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs) {
  PADDLE_ENFORCE(grad_op_descs_.empty(),
                 "Grad op descs must be initialized here");
  PADDLE_ENFORCE(preceding_ops_.empty(),
                 "Preceding ops must be initialized here");
  PADDLE_ENFORCE(bwd_ins_.empty(), "Backward inputs must be initialized here");
  PADDLE_ENFORCE(bwd_outs_.empty(),
                 "Backward outputs must be initialized here");

  // grad_to_var is a map of framework::GradVarName(in_var_name/out_var_name) ->
  // in_var_name/out_var_name
  std::unordered_map<std::string, std::string> grad_to_var;

  // NOTE(minqiyang): We don't support control flow op in imperative now
  // Add grad_block_ when we want to support it
  grad_op_descs_ = CreateGradOpDescs(op_->Info(), fwd_op, {}, {}, &grad_to_var);

  size_t grad_op_num = grad_op_descs_.size();

  VLOG(5) << "Create " << grad_op_num << " grad op desc(s) to op " << Type();

  if (grad_op_num == 0) return;

  // Build a map to record var_name -> std::shared_ptr<VarBase>*,
  // so that we can find suitable var in grad op descs
  std::unordered_map<std::string, const std::shared_ptr<VarBase>*> name_to_var;
  for (auto& pair : ins) {
    for (auto& var : pair.second) {
      auto& var_ptr = name_to_var[var->Name()];
      PADDLE_ENFORCE(var_ptr == nullptr || var_ptr->get() == var.get(),
                     "There are different variables with same name %s",
                     var->Name());
      var_ptr = &var;
    }
  }

  for (auto& pair : outs) {
    for (auto& var : pair.second) {
      auto& var_ptr = name_to_var[var->Name()];
      PADDLE_ENFORCE(var_ptr == nullptr || var_ptr->get() == var.get(),
                     "There are different variables with same name %s",
                     var->Name());
      var_ptr = &var;

      var->SetGeneratedOp(this);
    }
  }

  // Build backward ins and outs
  bwd_ins_.resize(grad_op_num);
  bwd_outs_.resize(grad_op_num);

  for (size_t i = 0; i < grad_op_num; ++i) {
    for (auto& grad_ins : grad_op_descs_[i]->Inputs()) {
      if (grad_ins.second.empty()) continue;
      auto& bwd_in = bwd_ins_[i][grad_ins.first];
      bwd_in.reserve(grad_ins.second.size());

      for (auto& grad_in_var_name : grad_ins.second) {
        auto iter = grad_to_var.find(grad_in_var_name);

        if (iter != grad_to_var.end()) {
          // If it is a grad var, find its coresponding forward var
          auto& fwd_var_name = iter->second;
          auto fwd_var_iter = name_to_var.find(fwd_var_name);
          PADDLE_ENFORCE(fwd_var_iter != name_to_var.end(),
                         "Cannot find forward variable named %s", fwd_var_name);

          bwd_in.emplace_back((*(fwd_var_iter->second))->GradVarBase());
        } else {
          // If it is a forward var, just add it
          auto fwd_var_iter = name_to_var.find(grad_in_var_name);
          PADDLE_ENFORCE(fwd_var_iter != name_to_var.end(),
                         "Cannot find forward variable named %s",
                         grad_in_var_name);
          bwd_in.emplace_back(*(fwd_var_iter->second));
        }

        VLOG(2) << "Set backward input " << grad_ins.first << " of " << Type()
                << " to be " << bwd_in.back()->Name();
      }
    }

    for (auto& grad_outs : grad_op_descs_[i]->Outputs()) {
      if (grad_outs.second.empty()) continue;
      auto& bwd_out = bwd_outs_[i][grad_outs.first];
      bwd_out.reserve(grad_outs.second.size());

      for (auto& grad_out_var_name : grad_outs.second) {
        auto iter = grad_to_var.find(grad_out_var_name);
        PADDLE_ENFORCE(iter != grad_to_var.end(),
                       "Cannot find output of input grad %s in op %s",
                       grad_out_var_name, Type());
        auto fwd_var_iter = name_to_var.find(iter->second);
        PADDLE_ENFORCE(fwd_var_iter != name_to_var.end(),
                       "Cannot find forward variable named %s", iter->second);
        bwd_out.emplace_back((*(fwd_var_iter->second))->GradVarBase());
        VLOG(2) << "Set backward output " << grad_outs.first << " of " << Type()
                << " to be "
                << (bwd_out.back() ? bwd_out.back()->Name() : "nullptr");

        auto* preceding_op = (*(fwd_var_iter->second))->GeneratedOp();

        if (preceding_op) {
          preceding_ops_.insert(preceding_op);
        }
      }
    }
  }
}

void OpBase::ClearBackwardTrace() {
  grad_op_descs_.clear();
  preceding_ops_.clear();
  bwd_ins_.clear();
  bwd_outs_.clear();
}

}  // namespace imperative
}  // namespace paddle
