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
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/rpc_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/scope.h"

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
  balance_vars_.resize(places_.size(), 0);
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

std::vector<std::string> MultiDevSSAGraphBuilder::FindDistTrainSendVars(
    const ProgramDesc &program) const {
  std::vector<std::string> send_vars;
  // since parameters are all in block 0,
  // it's enough to only scan send ops in block 0
  for (auto *op : program.Block(0).AllOps()) {
    // TODO(Yancey1989): use a graceful method to find send op,
    // instead of the the hard code string
    if (op->Type() == "send") {
      auto op_vars = op->InputArgumentNames();
      send_vars.reserve(send_vars.size() +
                        std::distance(op_vars.begin(), op_vars.end()));
      send_vars.insert(send_vars.end(), op_vars.begin(), op_vars.end());
    }
  }
  return send_vars;
}

std::vector<std::string> MultiDevSSAGraphBuilder::FindDistTrainRecvVars(
    const ProgramDesc &program) const {
  std::vector<std::string> recv_vars;
  for (auto *op : program.Block(0).AllOps()) {
    // TODO(Yancey1989): use a graceful method to find recv op,
    // instead of the hard code string
    if (op->Type() == "recv") {
      auto op_vars = op->OutputArgumentNames();
      recv_vars.reserve(recv_vars.size() +
                        std::distance(op_vars.begin(), op_vars.end()));
      recv_vars.insert(recv_vars.end(), op_vars.begin(), op_vars.end());
    }
  }
  return recv_vars;
}

bool MultiDevSSAGraphBuilder::IsDistTrainOp(
    const OpDesc &op, const std::vector<std::string> &send_vars,
    const std::vector<std::string> &recv_vars) const {
  if (send_vars.size() == 0 || recv_vars.size() == 0) {
    return false;
  }

  /**
   * Check any of opvars contains `.block` and in sendvars
   */
  auto checker = [](const std::vector<std::string> &opvars,
                    const std::vector<std::string> &rpc_vars) -> bool {
    for (auto &var : opvars) {
      // a variable name with the suffix `.block` means it's a splited
      // variable by (DistributeTranspiler)
      // [python/paddle/fluid/transpiler/distribute_transpiler.py]
      if (var.find(".block") != std::string::npos &&
          std::find(rpc_vars.begin(), rpc_vars.end(), var) != rpc_vars.end()) {
        return true;
      }
    }
    return false;
  };

  return checker(op.OutputArgumentNames(), send_vars) ||
         checker(op.InputArgumentNames(), recv_vars);
}

size_t MultiDevSSAGraphBuilder::GetAppropriateDeviceID(
    const std::vector<std::string> &var_names) const {
  int64_t numel_sum = 0;
  for (auto var_name : var_names) {
    auto var_desc = all_vars_.at(var_name);
    PADDLE_ENFORCE_NOT_NULL(var_desc);
    auto dim = framework::make_ddim(var_desc->GetShape());
    int64_t numel = framework::product(dim);
    PADDLE_ENFORCE_GT(numel, 0);
    numel_sum += numel;
  }

  auto smallest =
      std::min_element(std::begin(balance_vars_), std::end(balance_vars_));
  size_t dev_id =
      static_cast<size_t>(std::distance(std::begin(balance_vars_), smallest));
  balance_vars_[dev_id] += numel_sum;
  return dev_id;
}

std::unique_ptr<SSAGraph> MultiDevSSAGraphBuilder::Build(
    const ProgramDesc &program) const {
  for (auto *var : program.Block(0).AllVars()) {
    all_vars_.emplace(var->Name(), var);
  }

  auto graph = new SSAGraph();
  SSAGraph &result = *graph;
  std::unordered_set<std::string> og_has_been_broadcast;

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.vars_ = std::vector<
      std::unordered_map<std::string, std::vector<std::unique_ptr<VarHandle>>>>(
      places_.size());

  // find send/recv vars so that we can place the distributed training
  // realted op in the place 0
  auto send_vars = FindDistTrainSendVars(program);
  auto recv_vars = FindDistTrainRecvVars(program);

  std::vector<std::unordered_set<std::string>> bcast_var_name_set;
  bcast_var_name_set.resize(places_.size());

  size_t cur_device_id = 0;
  bool is_forwarding = true;

  for (auto *op : program.Block(0).AllOps()) {
    if (boost::get<int>(
            op->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
        static_cast<int>(OpRole::kRPC)) {
      CreateRPCOp(&result, *op);
    } else if (IsDistTrainOp(*op, send_vars, recv_vars)) {
      CreateDistTrainOp(&result, *op);
    } else if (IsScaleLossOp(*op)) {
      // user can customize loss@grad if not use_default_grad_scale_
      if (strategy_.gradient_scale_ !=
          BuildStrategy::GradientScaleStrategy::kCustomized) {
        CreateScaleLossGradOp(&result);
      }
      // This assumes the backward generating code will ensure IsScaleLossOp
      // is true only for the op that scale the final scalar loss.
      // It also assumes backward op will always follow the forward op in
      // the block.
      is_forwarding = false;
    } else {
      int op_dev_id = GetOpDeviceID(*op);
      if (op_dev_id == -1) {  // var on all device
        CreateComputationalOps(&result, *op, places_.size());
      } else {
        CreateComputationalOp(&result, *op, op_dev_id);
        for (auto &var_name : op->OutputArgumentNames()) {
          var_name_on_devices_.emplace(var_name, op_dev_id);
        }
      }
      if (!is_forwarding && places_.size() > 1) {
        // Currently, we assume that once gradient is generated, it can be
        // broadcast, and each gradient is only broadcast once.
        if (static_cast<bool>(boost::get<int>(op->GetAttr(
                                  OpProtoAndCheckerMaker::OpRoleAttrName())) &
                              static_cast<int>(OpRole::kBackward))) {
          try {
            auto backward_vars =
                boost::get<std::vector<std::string>>(op->GetNullableAttr(
                    OpProtoAndCheckerMaker::OpRoleVarAttrName()));

            PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);

            for (size_t i = 0; i < backward_vars.size(); i += 2) {
              auto &p_name = backward_vars[i];
              auto &g_name = backward_vars[i + 1];
              VLOG(10) << "Bcast " << g_name << " for parameter " << p_name;

              switch (strategy_.reduce_) {
                case BuildStrategy::ReduceStrategy::kReduce:
                  cur_device_id = GetAppropriateDeviceID({g_name});
                  CreateReduceOp(&result, g_name, cur_device_id);
                  var_name_on_devices_.emplace(g_name, cur_device_id);
                  bcast_var_name_set[cur_device_id].emplace(p_name);
                  break;
                case BuildStrategy::ReduceStrategy::kAllReduce:
                  if (IsSparseGradient(g_name)) {
                    CreateReduceOp(&result, g_name, 0);
                    CreateBroadcastOp(&result, g_name, 0);
                  } else {
                    InsertAllReduceOp(&result, g_name);
                  }
                  break;
                default:
                  LOG(FATAL) << "Unknown reduce strategy ";
                  break;
              }
            }
          } catch (boost::bad_get e) {
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
  /*
    Dependency graph has been constructed. However, there are still data
    hazards need to be handled.
   */
  PolishGraphToSupportDataHazards(&result);

  /*
   * Only variables should be the leaves of graph.
   */
  AddOutputToLeafOps(&result);

  return std::unique_ptr<SSAGraph>(graph);
}

bool MultiDevSSAGraphBuilder::IsSparseGradient(const std::string &og) const {
  PADDLE_ENFORCE(all_vars_.count(og) != 0);
  if (all_vars_.at(og)->GetType() == proto::VarType::SELECTED_ROWS) {
    return true;
  }
  return false;
}

void MultiDevSSAGraphBuilder::SetCommunicationContext(
    OpHandleBase *op_handle, const platform::Place &p) const {
#ifdef PADDLE_WITH_CUDA
  if (nccl_ctxs_ == nullptr) {
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#else
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));
#endif
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
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->vars_.at(i).at(p_name);
    auto *out_var = new VarHandle(vars.size(), i, p_name, p);
    vars.emplace_back(out_var);
    op_handle->AddOutput(out_var);
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOp(SSAGraph *result,
                                                    const OpDesc &op,
                                                    int dev_id) const {
  result->ops_.emplace_back(
      new ComputationOpHandle(op, local_scopes_[dev_id], places_[dev_id]));
  CreateOpHandleIOs(result, op, dev_id);
}

void MultiDevSSAGraphBuilder::InsertAllReduceOp(SSAGraph *result,
                                                const std::string &og) const {
#ifdef PADDLE_WITH_CUDA
  result->ops_.emplace_back(
      new AllReduceOpHandle(local_scopes_, places_, nccl_ctxs_));
#else
  result->ops_.emplace_back(new AllReduceOpHandle(local_scopes_, places_));
#endif
  auto *op_handle = result->ops_.back().get();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->vars_[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());

    auto var = new VarHandle(vars.size(), i, og, p);
    vars.emplace_back(var);
    op_handle->AddOutput(var);
  }
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

int MultiDevSSAGraphBuilder::GetOpDeviceID(const OpDesc &op) const {
  if (strategy_.reduce_ != BuildStrategy::ReduceStrategy::kReduce) {
    return -1;
  }

  for (auto &varname : op.InputArgumentNames()) {
    int dev_id = GetVarDeviceID(varname);
    if (dev_id != -1) {
      return dev_id;
    }
  }
  return -1;
}

int MultiDevSSAGraphBuilder::GetVarDeviceID(const std::string &varname) const {
  auto got = var_name_on_devices_.find(varname);
  return got == var_name_on_devices_.end() ? -1 : got->second;
}

void MultiDevSSAGraphBuilder::CreateScaleLossGradOp(SSAGraph *result) const {
  for (size_t i = 0; i < places_.size(); ++i) {
// Insert ScaleCost OpHandle
#ifdef PADDLE_WITH_CUDA
    auto *communication_dev_ctx =
        nccl_ctxs_ ? nccl_ctxs_->DevCtx(places_[i])
                   : platform::DeviceContextPool::Instance().Get(places_[i]);
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
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->vars_[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());
  }
  auto &vars = result->vars_[dst_dev_id][og];
  auto var = new VarHandle(vars.size(), dst_dev_id, og, places_[dst_dev_id]);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
  return var;
}

// Find the first occurence of `prev_op_name` and make current `op` depend
// on it.
void MultiDevSSAGraphBuilder::ConnectOp(SSAGraph *result, OpHandleBase *op,
                                        const std::string &prev_op_name) const {
  for (auto &prev_op : result->ops_) {
    if (prev_op->Name() == prev_op_name) {
      auto *dep_var = new DummyVarHandle();
      prev_op->AddOutput(dep_var);
      result->dep_vars_.emplace(dep_var);
      op->AddInput(dep_var);
    }
  }
}

void MultiDevSSAGraphBuilder::CreateDistTrainOp(SSAGraph *result,
                                                const OpDesc &op) const {
  int op_dev_id = -1;
  if (op.Type() == "split_byref" || op.Type() == "split_selected_rows") {
    op_dev_id = GetVarDeviceID(op.InputArgumentNames()[0]);
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
      op_dev_id = GetAppropriateDeviceID(op.InputArgumentNames());
      for (auto &varname : op.InputArgumentNames()) {
        var_name_on_devices_.emplace(varname, op_dev_id);
      }
    }
    for (auto &varname : op.OutputArgumentNames()) {
      var_name_on_devices_.emplace(varname, op_dev_id);
    }
  } else if (op.Type() == "concat") {
    op_dev_id = GetVarDeviceID(op.InputArgumentNames()[0]);
    for (auto &varname : op.OutputArgumentNames()) {
      var_name_on_devices_.emplace(varname, op_dev_id);
    }
  } else {
    PADDLE_ENFORCE(
        "the distribute training related op should be in [split_byref, "
        "concat].");
  }

  PADDLE_ENFORCE(op_dev_id != -1,
                 "can not find right place for distributed op: %s", op.Type());

  CreateComputationalOp(result, op, op_dev_id);
  if (op.Type() == "concat") {
    ConnectOp(result, result->ops_.back().get(), "fetch_barrier");
  }
}

// Create RPC related op handles that connects its in ops and out ops.
void MultiDevSSAGraphBuilder::CreateRPCOp(SSAGraph *result,
                                          const OpDesc &op) const {
  int op_dev_id = -1;
  if (op.Type() == "send") {
    op_dev_id = GetVarDeviceID(op.InputArgumentNames()[0]);
    // the variable name which contains .block means it was splited by
    // split_byref op
    // so that we can balance the variable blocks to all the pserver instances.
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce &&
        op.InputArgumentNames()[0].find(".block") == std::string::npos) {
      op_dev_id = GetAppropriateDeviceID(op.InputArgumentNames());
      for (auto &varname : op.InputArgumentNames()) {
        var_name_on_devices_.emplace(varname, op_dev_id);
      }
    }
  } else if (op.Type() == "recv") {
    op_dev_id = GetAppropriateDeviceID(op.OutputArgumentNames());
    for (auto &varname : op.OutputArgumentNames()) {
      var_name_on_devices_.emplace(varname, op_dev_id);
    }
  } else {
    // send_barrier and fetch_barrier op can be scheduled on device 0
    op_dev_id = 0;
  }

  PADDLE_ENFORCE(op_dev_id != -1, "can not find the right place for rpc op: %s",
                 op.Type());

  result->ops_.emplace_back(new RPCOpHandle(op, local_scopes_[op_dev_id],
                                            op.Type(), places_[op_dev_id]));

  if (op.Type() == "send_barrier") {
    ConnectOp(result, result->ops_.back().get(), "send");
  } else if (op.Type() == "recv") {
    ConnectOp(result, result->ops_.back().get(), "send_barrier");
  } else if (op.Type() == "fetch_barrier") {
    ConnectOp(result, result->ops_.back().get(), "recv");
  } else if (op.Type() == "send") {
    // do nothing
  } else {
    PADDLE_THROW(
        "rpc op should be in ["
        "send, send_barrier. recv, fetch_barrier]");
  }

  CreateOpHandleIOs(result, op, op_dev_id);
}

bool MultiDevSSAGraphBuilder::IsScaleLossOp(const OpDesc &op) const {
  return boost::get<int>(
             op.GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
             (static_cast<int>(OpRole::kBackward) |
              static_cast<int>(OpRole::kLoss)) &&
         !loss_var_name_.empty();  // If loss_var is empty. This is test mode
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
