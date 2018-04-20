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
    platform::NCCLContextMap *nccl_ctxs, bool use_nccl_allreduce)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes),
      nccl_ctxs_(nccl_ctxs),
      use_nccl_allreduce_(use_nccl_allreduce) {
#else

MultiDevSSAGraphBuilder::MultiDevSSAGraphBuilder(
    const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &params,
    const std::vector<Scope *> &local_scopes, bool use_nccl_allreduce)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes),
      use_nccl_allreduce_(use_nccl_allreduce) {
#endif
  for (auto &p : params) {
    grad_names_.insert(GradVarName(p));
  }
}

// void GetLeavesOfTopVar(VarLink *top_var,
//                       std::unordered_set<VarHandle *> *leaves) {
//  if (top_var && top_var->children_.size() == 0) {
//    leaves->insert(top_var->var_handle_);
//  } else if (top_var) {
//    for (auto &vars : top_var->children_) {
//      GetLeavesOfTopVar(vars, leaves);
//    }
//  }
//}

void MultiDevSSAGraphBuilder::CreateOpHandleIOs(SSAGraph *result,
                                                const OpDesc &op,
                                                const platform::Place &p,
                                                const size_t &i) const {
  auto *op_handle = result->ops_.back().get();
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));

  auto var_names = op.InputArgumentNames();

  for (auto &each_var_name : var_names) {
    VarHandle *var = CreateOrGetLatestVarHandle(result, each_var_name, p, i);
    op_handle->AddInput(var);
  }

  var_names = op.OutputArgumentNames();

  for (auto &each_var_name : var_names) {
    CreateOpOutput(result, op_handle, each_var_name, p, i);
  }
}

std::unique_ptr<SSAGraph> MultiDevSSAGraphBuilder::Build(
    const ProgramDesc &program) const {
  auto graph = new SSAGraph();
  SSAGraph &result = *graph;
  std::unordered_set<std::string> og_has_been_broadcast;

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.vars_ = std::vector<
      std::unordered_map<std::string, std::vector<std::unique_ptr<VarHandle>>>>(
      places_.size());

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
    } else if (IsScaleLossOp(*op)) {
      CreateScaleLossGradOp(&result);
      is_forwarding = false;
    } else {
      int op_dev_id = GetOpDeviceID(var_name_on_devices, *op);
      if (op_dev_id == -1) {  // var on all device
        CreateComputationalOps(&result, *op);
      } else {
        CreateComputationalOp(&result, *op, op_dev_id);
        for (auto &var_name : op->OutputArgumentNames()) {
          var_name_on_devices[op_dev_id].emplace(var_name);
        }
      }

      if (!is_forwarding && places_.size() > 1) {
        // Currently, we assume that once gradient is generated, it can be
        // broadcast, and each gradient is only broadcast once. But there are no
        // other cases, for example, we need to adjust the gradient according to
        // the input when we get the gradient, which is not considered at
        // present.
        for (auto &og : op->OutputArgumentNames()) {
          if (IsParameterGradientOnce(og, &og_has_been_broadcast)) {
            if (use_nccl_allreduce_) {
              InsertNCCLAllReduceOp(&result, og);
            } else {
              CreateReduceOp(&result, cur_device_id, og);
              var_name_on_devices[cur_device_id].emplace(og);
              bcast_var_name_set[cur_device_id].emplace(
                  og.substr(0, og.size() - strlen(kGradVarSuffix)));
              cur_device_id = (cur_device_id + 1) % places_.size();
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

int MultiDevSSAGraphBuilder::GetOpDeviceID(
    const std::vector<std::unordered_set<std::string>> &var_name_on_devices,
    const OpDesc &op) const {
  if (use_nccl_allreduce_) {
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

void MultiDevSSAGraphBuilder::CreateBroadcastOp(SSAGraph *result,
                                                const std::string &p_name,
                                                size_t dev_id) const {
  auto *op_handle = new BroadcastOpHandle(local_scopes_, places_);
  result->ops_.emplace_back(op_handle);
  auto *in = result->vars_.at(dev_id).at(p_name).back().get();
  op_handle->AddInput(in);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &vars = result->vars_.at(dev_id).at(p_name);
    auto &p = places_[i];
    auto *out_var = new VarHandle(vars.size(), i, p_name, p);
    vars.emplace_back(out_var);
    op_handle->AddOutput(out_var);

#ifdef PADDLE_WITH_CUDA
    op_handle->SetDeviceContext(p, nccl_ctxs_->DevCtx(p));
#else
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
  CreateOpHandleIOs(result, op, places_[dev_id], dev_id);
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
                                                     const OpDesc &op) const {
  for (size_t scope_idx = 0; scope_idx < places_.size(); ++scope_idx) {
    auto p = places_[scope_idx];
    auto s = local_scopes_[scope_idx];
    result->ops_.emplace_back(new ComputationOpHandle(op, s, p));
    CreateOpHandleIOs(result, op, p, scope_idx);
  }
}

VarHandle *MultiDevSSAGraphBuilder::CreateReduceOp(
    SSAGraph *result, int dst_dev_id, const std::string &og) const {
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
  CreateOpHandleIOs(result, op, p, 0);
}

bool MultiDevSSAGraphBuilder::IsScaleLossOp(const OpDesc &op) const {
  // FIXME(yy): Do not hard code like this
  return op.OutputArgumentNames().size() == 1 &&
         op.OutputArgumentNames()[0] == GradVarName(loss_var_name_);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
