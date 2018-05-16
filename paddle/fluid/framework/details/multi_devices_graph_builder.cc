//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include <utility>
#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/details/send_op_handle.h"
#include "paddle/fluid/framework/scope.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"
#endif

#include <string>
#include <vector>

namespace paddle {
namespace framework {
namespace details {

#ifdef PADDLE_WITH_CUDA
MultiDevSSAGraphBuilder::MultiDevSSAGraphBuilder(
    const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &params,
    const std::vector<Scope *> &local_scopes,
    platform::NCCLContextMap *nccl_ctxs, const BuildStrategy &strategy)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes),
      nccl_ctxs_(nccl_ctxs),
      strategy_(strategy) {
#else
MultiDevSSAGraphBuilder::MultiDevSSAGraphBuilder(
    const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &params,
    const std::vector<Scope *> &local_scopes, const BuildStrategy &strategy)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes),
      strategy_(strategy) {
#endif
  for (auto &p : params) {
    grad_names_.insert(GradVarName(p));
  }
}

void MultiDevSSAGraphBuilder::CreateOpHandleIOs(SSAGraph *result,
                                                const OpDesc &op,
                                                size_t place_id) const {
  auto p = places_[place_id];
  auto *op_handle = result->ops_.back().get();
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));

  for (auto &each_var_name : op.InputArgumentNames()) {
    VarHandle *var =
        CreateOrGetLatestVarHandle(result, each_var_name, p, place_id);
    op_handle->AddInput(var);
  }

  for (auto &each_var_name : op.OutputArgumentNames()) {
    CreateOpOutput(result, op_handle, each_var_name, p, place_id);
  }
}

bool MultiDevSSAGraphBuilder::IsDistTrainOp(const OpDesc &op,
                                            OpDesc *send_op) const {
  if (send_op == nullptr) {
    return false;
  }

  /**
   * Check any of opvars contains `.block` and in sendvars
   */
  auto checker = [](const std::vector<std::string> &opvars,
                    const std::vector<std::string> &sendvars) -> bool {
    for (auto &var : opvars) {
      if (var.find(".block") != std::string::npos &&
          std::find(sendvars.begin(), sendvars.end(), var) != sendvars.end()) {
        return true;
      }
    }
    return false;
  };

  if (op.Type() == "split") {
    return checker(op.OutputArgumentNames(), send_op->InputArgumentNames());
  } else if (op.Type() == "concat") {
    return checker(op.InputArgumentNames(), send_op->OutputArgumentNames());
  }
  return false;
}

std::unique_ptr<SSAGraph> MultiDevSSAGraphBuilder::Build(
    const ProgramDesc &program) const {
  std::unordered_map<std::string, VarDesc *> all_vars;
  for (auto *var : program.Block(0).AllVars()) {
    all_vars[var->Name()] = var;
  }

  auto graph = new SSAGraph();
  SSAGraph &result = *graph;
  std::unordered_set<std::string> og_has_been_broadcast;

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.vars_ = std::vector<
      std::unordered_map<std::string, std::vector<std::unique_ptr<VarHandle>>>>(
      places_.size());

  // Find "send" op first for split is in front of send.
  OpDesc *send_op = GetSendOpDesc(program);

  size_t cur_device_id = 0;
  std::vector<std::unordered_set<std::string>> var_name_on_devices;
  std::vector<std::unordered_set<std::string>> bcast_var_name_set;
  var_name_on_devices.resize(places_.size());
  bcast_var_name_set.resize(places_.size());

  bool is_forwarding = true;
  for (auto *op : program.Block(0).AllOps()) {
    if (op->Type() == "send") {
      // append send op if program is distributed trainer main program.
      // always use the first device
      CreateSendOp(&result, *op);
    } else if (IsDistTrainOp(*op, send_op)) {
      CreateComputationalOps(&result, *op, 1);
    } else if (IsScaleLossOp(*op)) {
      // user can customize loss@grad if not use_default_grad_scale_
      if (strategy_.gradient_scale_ !=
          BuildStrategy::GradientScaleStrategy::kCustomized) {
        CreateScaleLossGradOp(&result);
      }
      is_forwarding = false;
    } else {
      int op_dev_id = GetOpDeviceID(var_name_on_devices, *op);
      if (op_dev_id == -1) {  // var on all device
        CreateComputationalOps(&result, *op, places_.size());
      } else {
        CreateComputationalOp(&result, *op, op_dev_id);
        for (auto &var_name : op->OutputArgumentNames()) {
          var_name_on_devices[op_dev_id].emplace(var_name);
        }
      }
      if (!is_forwarding && places_.size() > 1) {
        // Currently, we assume that once gradient is generated, it can be
        // broadcast, and each gradient is only broadcast once.
        for (auto &og : op->OutputArgumentNames()) {
          if (IsParameterGradientOnce(og, &og_has_been_broadcast)) {
            switch (strategy_.reduce_) {
              case BuildStrategy::ReduceStrategy::kReduce:
                CreateReduceOp(&result, og, cur_device_id);
                var_name_on_devices[cur_device_id].emplace(og);
                bcast_var_name_set[cur_device_id].emplace(
                    og.substr(0, og.size() - strlen(kGradVarSuffix)));
                cur_device_id = (cur_device_id + 1) % places_.size();
                break;
              case BuildStrategy::ReduceStrategy::kAllReduce:
                if (IsSparseGradient(all_vars, og)) {
                  CreateReduceOp(&result, og, 0);
                  CreateBroadcastOp(&result, og, 0);
                } else {
                  InsertNCCLAllReduceOp(&result, og);
                }
                break;
            }
          }
        }
      }
    }
  }

  // Insert BCast Ops
  for (size_t dev_id = 0; dev_id < bcast_var_name_set.size(); ++dev_id) {
    auto &to_bcast_set = bcast_var_name_set[dev_id];
    for (auto &bcast_name : to_bcast_set) {
      CreateBroadcastOp(&result, bcast_name, dev_id);
    }
  }

  // Fuse reduce operation
  FuseReduceOpHandles(all_vars, &result);

  /*
    Dependency graph has been constructed. However, there are still data
    harzaeds need to be handled.
   */
  PolishGraphToSupportDataHazards(&result);

  /*
   * Only variables should be the leaves of graph.
   */
  AddOutputToLeafOps(&result);

  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    PrintGraphviz(*graph, sout);
    VLOG(10) << sout.str();
  }

  return std::unique_ptr<SSAGraph>(graph);
}

// Fuse reduce operation
void MultiDevSSAGraphBuilder::FuseReduceOpHandles(
    const std::unordered_map<std::string, VarDesc *> &all_vars,
    SSAGraph *result) const {
  std::vector<std::unordered_set<OpHandleBase *>> reduce_op_handles;
  std::vector<std::unordered_set<VarHandle *>> input_set_of_reduce_op;
  std::vector<std::unordered_set<VarHandle *>> output_set_of_reduce_op;
  reduce_op_handles.resize(this->places_.size());
  input_set_of_reduce_op.resize(this->places_.size());
  output_set_of_reduce_op.resize(this->places_.size());

  bool has_reduce_op = false;
  // First, collect all the reduce_ops
  // Only fuse the reduce_ops whose outputs' type is proto::VarType::LOD_TENSOR
  // and output's place is on GPU .
  for (size_t i = 0; i < result->ops_.size(); ++i) {
    ReduceOpHandle *reduce_op =
        dynamic_cast<ReduceOpHandle *>(result->ops_[i].get());
    if (reduce_op) {
      auto out_var_handle = dynamic_cast<VarHandle *>(reduce_op->Outputs()[0]);
      PADDLE_ENFORCE_NOT_NULL(out_var_handle);
      if (all_vars.at(out_var_handle->name_)->GetType() !=
              proto::VarType::LOD_TENSOR ||
          platform::is_cpu_place(out_var_handle->place_)) {
        continue;
      }
      size_t dev_id = out_var_handle->scope_idx_;
      reduce_op_handles.at(dev_id).insert(result->ops_[i].get());
      for (auto in : reduce_op->Inputs()) {
        input_set_of_reduce_op.at(dev_id).insert(dynamic_cast<VarHandle *>(in));
      }
      output_set_of_reduce_op.at(dev_id).insert(out_var_handle);
      has_reduce_op = true;
    }
  }
  if (!has_reduce_op) return;

  // Second, alloc continuous address and create new reduce_op
  for (size_t dev_id = 0; dev_id < reduce_op_handles.size(); ++dev_id) {
    // AllocateContinuousAddress
    auto out_var_handles = output_set_of_reduce_op[dev_id];
    if (out_var_handles.size() == 0) continue;

    platform::Place run_place = (*out_var_handles.begin())->place_;
    proto::VarType::Type out0_var_data_type =
        all_vars.at((*out_var_handles.begin())->name_)->GetDataType();

    int64_t total_numel = 0;
    for (auto out_var_handle : out_var_handles) {
      auto var_desc = all_vars.at(out_var_handle->name_);

      PADDLE_ENFORCE_EQ(out0_var_data_type, var_desc->GetDataType());
      PADDLE_ENFORCE_EQ(run_place, out_var_handle->place_);

      auto dim = framework::make_ddim(var_desc->GetShape());
      int64_t numel = framework::product(dim);
      PADDLE_ENFORCE_GT(numel, 0);
      total_numel += numel;
    }

    auto reduce_var_name = string::Sprintf("REDUCEBLOCK_DATA_%d", dev_id);
    for (size_t k = 0; k < this->places_.size(); ++k) {
      // Allocate gradients memory
      auto reduce_t = this->local_scopes_.at(k)
                          ->Var(reduce_var_name)
                          ->GetMutable<LoDTensor>();
      PADDLE_ENFORCE(platform::is_gpu_place(this->places_[k]));

      reduce_t->Resize(make_ddim({total_numel}))
          .mutable_data(this->places_[k],
                        framework::ToTypeIndex(out0_var_data_type));

      VLOG(8) << this->places_[k] << " " << reduce_var_name
              << " total_numel: " << total_numel;

      int64_t s = 0;
      for (auto out : out_var_handles) {
        LoDTensor *tensor =
            this->local_scopes_.at(k)->Var(out->name_)->GetMutable<LoDTensor>();

        int64_t mem_size = framework::product(
            framework::make_ddim(all_vars.at(out->name_)->GetShape()));
        tensor->ShareDataWith(reduce_t->Slice(s, s + mem_size));

        VLOG(8) << out->name_ << " mem_size:" << mem_size << " s:" << s
                << " e:" << s + mem_size;

        s += mem_size;
      }
    }
    CreateReduceBlockOp(result, dev_id, reduce_var_name,
                        input_set_of_reduce_op.at(dev_id),
                        output_set_of_reduce_op.at(dev_id));
  }
  RemoveOps(reduce_op_handles, result);
}

void MultiDevSSAGraphBuilder::RemoveOps(
    const std::vector<std::unordered_set<OpHandleBase *>> &reduce_op_handles,
    SSAGraph *result) const {
  for (auto iter = result->ops_.begin(); iter != result->ops_.end();) {
    bool remove_op = false;
    for (size_t j = 0; j < reduce_op_handles.size(); ++j) {
      if (reduce_op_handles[j].count((*iter).get())) {
        for (auto in_var : (*iter)->Inputs()) {
          in_var->pending_ops_.erase(in_var->pending_ops_.find((*iter).get()));
        }
        iter = result->ops_.erase(iter);
        remove_op = true;
        break;
      }
    }
    if (!remove_op) {
      iter++;
    }
  }
}

void MultiDevSSAGraphBuilder::CreateReduceBlockOp(
    SSAGraph *result, const int dst_scope_id,
    const std::string &reduce_var_name,
    const std::unordered_set<VarHandle *> &inputs,
    const std::unordered_set<VarHandle *> &outputs) const {
#ifdef PADDLE_WITH_CUDA
  result->ops_.emplace_back(new ReduceOpHandle(
      local_scopes_, places_, nccl_ctxs_, dst_scope_id, reduce_var_name));
#else
  result->ops_.emplace_back(new ReduceOpHandle(local_scopes_, places_,
                                               dst_scope_id, reduce_var_name));
#endif
  auto *op_handle = result->ops_.back().get();

#ifndef PADDLE_WITH_CUDA
  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#endif

  for (auto input : inputs) {
    op_handle->AddInput(input);
  }
  for (auto output : outputs) {
    op_handle->AddOutput(output);
  }
}

bool MultiDevSSAGraphBuilder::IsSparseGradient(
    const std::unordered_map<std::string, VarDesc *> &all_vars,
    const std::string &og) const {
  PADDLE_ENFORCE(all_vars.count(og) != 0);
  if (all_vars.at(og)->GetType() == proto::VarType::SELECTED_ROWS) {
    return true;
  }
  return false;
}

void MultiDevSSAGraphBuilder::CreateBroadcastOp(SSAGraph *result,
                                                const std::string &p_name,
                                                size_t src_dev_id) const {
#ifdef PADDLE_WITH_CUDA
  auto *op_handle = new BroadcastOpHandle(local_scopes_, places_, nccl_ctxs_);
#else
  auto *op_handle = new BroadcastOpHandle(local_scopes_, places_);
#endif

  result->ops_.emplace_back(op_handle);
  auto *in = result->vars_.at(src_dev_id).at(p_name).back().get();
  op_handle->AddInput(in);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &vars = result->vars_.at(i).at(p_name);
    auto &p = places_[i];
    auto *out_var = new VarHandle(vars.size(), i, p_name, p);
    vars.emplace_back(out_var);
    op_handle->AddOutput(out_var);
#ifndef ADDLE_WITH_CUDA
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
#endif
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOp(SSAGraph *result,
                                                    const OpDesc &op,
                                                    int dev_id) const {
  result->ops_.emplace_back(
      new ComputationOpHandle(op, local_scopes_[dev_id], places_[dev_id]));
  CreateOpHandleIOs(result, op, dev_id);
}

OpDesc *MultiDevSSAGraphBuilder::GetSendOpDesc(
    const ProgramDesc &program) const {
  for (auto *op : program.Block(0).AllOps()) {
    if (op->Type() == "send") {
      return op;
    }
  }
  return nullptr;
}
void MultiDevSSAGraphBuilder::InsertNCCLAllReduceOp(
    SSAGraph *result, const std::string &og) const {
#ifdef PADDLE_WITH_CUDA
  result->ops_.emplace_back(
      new NCCLAllReduceOpHandle(local_scopes_, places_, *nccl_ctxs_));
  auto *op_handle = result->ops_.back().get();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    auto &vars = result->vars_[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());

    auto var = new VarHandle(vars.size() - 1, i, og, p);
    vars.emplace_back(var);
    op_handle->AddOutput(var);
  }
#else
  PADDLE_ENFORCE("Not implemented");
#endif
}

bool MultiDevSSAGraphBuilder::IsParameterGradientOnce(
    const std::string &og,
    std::unordered_set<std::string> *og_has_been_broadcast) const {
  bool is_pg_once =
      grad_names_.count(og) != 0 && og_has_been_broadcast->count(og) == 0;
  if (is_pg_once) {
    // Insert NCCL AllReduce Op
    og_has_been_broadcast->insert(og);
  }
  return is_pg_once;
}

int MultiDevSSAGraphBuilder::GetOpDeviceID(
    const std::vector<std::unordered_set<std::string>> &var_name_on_devices,
    const OpDesc &op) const {
  if (strategy_.reduce_ != BuildStrategy::ReduceStrategy::kReduce) {
    return -1;
  }

  int var_dev_id = -1;
  for (auto &var_name : op.InputArgumentNames()) {
    if (var_dev_id != -1) break;
    for (size_t i = 0; i < var_name_on_devices.size(); ++i) {
      if (var_name_on_devices[i].count(var_name)) {
        var_dev_id = static_cast<int>(i);
        break;
      }
    }
  }
  return var_dev_id;
}

void MultiDevSSAGraphBuilder::CreateScaleLossGradOp(SSAGraph *result) const {
  for (size_t i = 0; i < places_.size(); ++i) {
// Insert ScaleCost OpHandle
#ifdef PADDLE_WITH_CUDA
    auto *communication_dev_ctx = nccl_ctxs_->DevCtx(places_[i]);
#else
    auto *communication_dev_ctx =
        platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
#endif

    auto *op_handle =
        new ScaleLossGradOpHandle(local_scopes_.size(), local_scopes_[i],
                                  places_[i], communication_dev_ctx);
    result->ops_.emplace_back(op_handle);

    // FIXME: Currently ScaleLossGradOp only use device_count as scale
    // factor. So it does not depend on any other operators.
    // VarHandle *loss = GetVarHandle(loss_var_name, place);
    // loss->pending_ops_.emplace_back(op_handle);
    // op_handle->inputs_.emplace_back(loss);

    CreateOpOutput(result, op_handle, GradVarName(loss_var_name_), places_[i],
                   i);
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOps(SSAGraph *result,
                                                     const OpDesc &op,
                                                     size_t num_places) const {
  for (size_t scope_idx = 0; scope_idx < num_places; ++scope_idx) {
    auto p = places_[scope_idx];
    auto s = local_scopes_[scope_idx];
    result->ops_.emplace_back(new ComputationOpHandle(op, s, p));
    CreateOpHandleIOs(result, op, scope_idx);
  }
}

VarHandle *MultiDevSSAGraphBuilder::CreateReduceOp(SSAGraph *result,
                                                   const std::string &og,
                                                   int dst_dev_id) const {
#ifdef PADDLE_WITH_CUDA
  result->ops_.emplace_back(
      new ReduceOpHandle(local_scopes_, places_, nccl_ctxs_));
#else
  result->ops_.emplace_back(new ReduceOpHandle(local_scopes_, places_));
#endif
  auto *op_handle = result->ops_.back().get();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &vars = result->vars_[i][og];
#ifndef PADDLE_WITH_CUDA
    auto &p = places_[i];
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
#endif
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());
  }
  auto &vars = result->vars_[dst_dev_id][og];
  auto var =
      new VarHandle(vars.size() - 1, dst_dev_id, og, places_[dst_dev_id]);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
  return var;
}

void MultiDevSSAGraphBuilder::CreateSendOp(SSAGraph *result,
                                           const OpDesc &op) const {
  auto &p = places_[0];
  auto *s = local_scopes_[0];
  // FIXME(wuyi): send op always copy from GPU 0
  result->ops_.emplace_back(new SendOpHandle(op, s, p));
  // Create inputs for output on original place and no ssa output
  // is created for send op.
  CreateOpHandleIOs(result, op, 0);
}

bool MultiDevSSAGraphBuilder::IsScaleLossOp(const OpDesc &op) const {
  // FIXME(yy): Do not hard code like this
  return op.OutputArgumentNames().size() == 1 &&
         op.OutputArgumentNames()[0] == GradVarName(loss_var_name_);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
