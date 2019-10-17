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

using framework::Variable;
void ThreadSafeNameSet::Insert(const std::string& name) {
  std::lock_guard<std::mutex> guard(mtx_);
  set_.insert(name);
}

void ThreadSafeNameSet::Remove(const std::string& name) {
  std::lock_guard<std::mutex> guard(mtx_);
  auto iter = set_.find(name);
  PADDLE_ENFORCE_EQ(iter != set_.end(), true, "%s does not exist", name);
  set_.erase(iter);
}

std::vector<std::string> ThreadSafeNameSet::Names() const {
  std::lock_guard<std::mutex> guard(mtx_);
  return std::vector<std::string>(set_.begin(), set_.end());
}

ThreadSafeNameSet VarBase::name_set_;

std::vector<std::string> VarBase::AliveVarNames() { return name_set_.Names(); }

static framework::VariableNameMap CreateVarNameMap(
    const framework::OpInfo& op_info, const std::string& op_type,
    const NameVarBaseMap& varbase_map, bool is_input) {
  if (op_info.proto_ == nullptr) {
    return {};
  }

  framework::VariableNameMap result;

  for (auto& var :
       is_input ? op_info.Proto().inputs() : op_info.Proto().outputs()) {
    auto it = varbase_map.find(var.name());
    if (it == varbase_map.end()) {
      PADDLE_ENFORCE_EQ(
          var.dispensable(), true,
          "Var: %s not dispensable and there are no such var in inputs",
          var.name());
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

std::string LayerDebugString(const std::string& op_type,
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

void VarBase::AddGradOps(const std::weak_ptr<OpBase>& op) {
  if (op.lock() == nullptr) {
    return;
  }
  for (const auto& cur_op : grad_ops_) {
    if (cur_op.lock() == op.lock()) {
      return;
    }
  }
  grad_ops_.emplace_back(op);
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
  PADDLE_ENFORCE_EQ(var_.IsInitialized() && var_.IsType<framework::LoDTensor>(),
                    true,
                    "Variable must be initialized and type of LoDTensor when "
                    "getting numpy tensor");

  auto& src_tensor = var_.Get<framework::LoDTensor>();

  // TODO(Jiabin): change this after move unique_name generator to CXX
  auto new_var = std::make_shared<VarBase>(
      false, "Itmp" + std::to_string(copied_counter_++));

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
// create OpBase from optype
OpBase::OpBase(size_t id, const std::string& type, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, framework::AttributeMap attrs,
               const platform::Place& place)
    : id_(id), place_(place) {
  const auto& info = framework::OpInfoMap::Instance().Get(type);

  // Step 1: Run forward
  if (info.Checker() != nullptr) {
    info.Checker()->Check(&attrs);
  }

  auto input_name_map = CreateVarNameMap(info, type, ins, true);
  auto output_name_map = CreateVarNameMap(info, type, outs, false);
  op_ = framework::OpRegistry::CreateOp(type, std::move(input_name_map),
                                        std::move(output_name_map),
                                        std::move(attrs));
  VLOG(3) << "Construct Op: " << type << std::endl;
}

// create OpBase from opdesc
OpBase::OpBase(size_t id, const framework::OpDesc& op_desc,
               const platform::Place& place)
    : id_(id), op_(framework::OpRegistry::CreateOp(op_desc)), place_(place) {
  VLOG(3) << "Construct Op: " << op_desc.Type() << std::endl;
}

void OpBase::Run(const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  auto* op_kernel = dynamic_cast<framework::OperatorWithKernel*>(op_.get());
  PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");
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
  VLOG(5) << LayerDebugString(Type(), ins, outs);
  auto runtime_ctx = PrepareRuntimeContext(ins, outs);

  VLOG(6) << "start preparing op: " << Type();
  auto prepared_op = PreparedOp::Prepare(runtime_ctx, *op_kernel, place(), ins);

  VLOG(6) << "finish preparing op: " << Type();
  prepared_op.Run();

  VLOG(4) << LayerDebugString(Type(), ins, outs);
}

void OpBase::ClearBackwardTrace() {
  grad_pending_ops_.clear();
  ins_.clear();
  outs_.clear();
}

}  // namespace imperative
}  // namespace paddle
