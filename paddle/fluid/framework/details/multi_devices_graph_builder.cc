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
#include "paddle/fluid/framework/details/data_balance_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/rpc_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/ir/node.h"
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
  if (strategy_.enable_data_balance_ && places_.size() == 1) {
    LOG(WARNING) << "It is no need to enable data balance when there is only "
                    "one place. enable_data_balance is set to False.";
    strategy_.enable_data_balance_ = false;
  }
}

void MultiDevSSAGraphBuilder::CreateOpHandleIOs(Graph *result, ir::Node *node,
                                                size_t place_id) const {
  auto p = places_[place_id];
  auto *op_handle = result->Get<GraphOps>("ops").back().get();
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));

  for (ir::Node *input : node->inputs) {
    VarHandle *var = CreateOrGetLatestVarHandle(result, input, p, place_id);
    op_handle->AddInput(var);
  }

  for (ir::Node *output : node->outputs) {
    ir::Node *new_node = nullptr;
    if (output->Var()) {
      new_node = result->CreateVarNode(output->Var());
    } else {
      new_node =
          result->CreateEmptyNode(output->Name(), ir::Node::Type::kVariable);
    }
    CreateOpOutput(result, op_handle, new_node, p, place_id);
  }
}

std::vector<std::string> MultiDevSSAGraphBuilder::FindDistTrainSendVars(
    const std::vector<std::unique_ptr<ir::Node>> &nodes) const {
  std::vector<std::string> send_vars;
  // since parameters are all in block 0,
  // it's enough to only scan send ops in block 0
  for (auto &node : nodes) {
    if (node->NodeType() != ir::Node::Type::kOperation) continue;
    OpDesc *op = node->Op();
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
    const std::vector<std::unique_ptr<ir::Node>> &nodes) const {
  std::vector<std::string> recv_vars;
  for (auto &node : nodes) {
    if (node->NodeType() != ir::Node::Type::kOperation) continue;
    OpDesc *op = node->Op();
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
    ir::Node *node, const std::vector<std::string> &send_vars,
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

  std::vector<std::string> input_var_names;
  std::vector<std::string> output_var_names;
  for (ir::Node *input : node->inputs) {
    input_var_names.push_back(input->Name());
  }
  for (ir::Node *output : node->outputs) {
    output_var_names.push_back(output->Name());
  }

  return checker(output_var_names, send_vars) ||
         checker(input_var_names, recv_vars);
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

std::unique_ptr<Graph> MultiDevSSAGraphBuilder::Apply(
    std::unique_ptr<Graph> graph) const {
  // Rebuild the graph structure.
  auto nodes = std::move(graph->nodes);
  graph->nodes.clear();

  for (auto &node : nodes) {
    if (node->NodeType() == ir::Node::Type::kVariable) {
      all_vars_.emplace(node->Name(), node->Var());
    }
  }

  Graph &result = *graph;
  std::unordered_set<std::string> og_has_been_broadcast;

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.Set("vars", new GraphVars(places_.size()));
  result.Set("dep_vars", new GraphDepVars);
  result.Set("ops", new GraphOps);

  // find send/recv vars so that we can place the distributed training
  // realted op in the place 0
  auto send_vars = FindDistTrainSendVars(nodes);
  auto recv_vars = FindDistTrainRecvVars(nodes);

  std::vector<std::unordered_set<std::string>> bcast_var_name_set;
  bcast_var_name_set.resize(places_.size());

  size_t cur_device_id = 0;
  bool is_forwarding = true;

  // NOTE: Currently, passes before SSAGraphBuilder cannot reorder
  // forward, backward nodes. E.g. you can't append an forward node
  // at the end of the node list.
  // TODO(panyx0718): FIXME: Needs to sort by forward->backward order.
  for (auto &node : nodes) {
    if (node->NodeType() != ir::Node::Type::kOperation) continue;
    if (boost::get<int>(
            node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
        static_cast<int>(OpRole::kRPC)) {
      CreateRPCOp(&result, node.get());
    } else if (IsDistTrainOp(node.get(), send_vars, recv_vars)) {
      CreateDistTrainOp(&result, node.get());
    } else if (IsScaleLossOp(node.get())) {
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
      int op_dev_id = GetOpDeviceID(node.get());
      if (op_dev_id != -1) {  // This op only runs on one specific device.
        CreateComputationalOp(&result, node.get(), op_dev_id);
        for (ir::Node *n : node->outputs) {
          var_name_on_devices_.emplace(n->Name(), op_dev_id);
        }
      } else {
        // This op runs on all devices, and its output may have parameter's
        // gradients.
        if (node->Op()->Type() == "read" && strategy_.enable_data_balance_) {
          node->Op()->SetAttr("throw_eof_exp", false);
          CreateComputationalOps(&result, node.get(), places_.size());
          // TODO(paddle-dev): builder shouldn't depend on the out logic of
          // a specific op.
          const auto &data_var_names = node->Op()->Output("Out");
          InsertDataBalanceOp(&result, data_var_names);
        } else {
          CreateComputationalOps(&result, node.get(), places_.size());
        }

        if (!is_forwarding && places_.size() > 1) {
          // Currently, we assume that once gradient is generated, it can be
          // broadcast, and each gradient is only broadcast once.
          if (static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                    OpProtoAndCheckerMaker::OpRoleAttrName())) &
                                static_cast<int>(OpRole::kBackward))) {
            try {
              auto backward_vars = boost::get<std::vector<std::string>>(
                  node->Op()->GetNullableAttr(
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
  }

  bool use_gpu = false;
#ifdef PADDLE_WITH_CUDA
  use_gpu = nccl_ctxs_ != nullptr;
#endif

  if (use_gpu ||
      strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
    // Insert BCast Ops
    for (size_t dev_id = 0; dev_id < bcast_var_name_set.size(); ++dev_id) {
      auto &to_bcast_set = bcast_var_name_set[dev_id];
      for (auto &bcast_name : to_bcast_set) {
        CreateBroadcastOp(&result, bcast_name, dev_id);
      }
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
  return std::move(graph);
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

void MultiDevSSAGraphBuilder::CreateBroadcastOp(Graph *result,
                                                const std::string &p_name,
                                                size_t src_dev_id) const {
#ifdef PADDLE_WITH_CUDA
  auto *op_handle = new BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_);
#else
  auto *op_handle = new BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_);
#endif
  result->Get<GraphOps>("ops").emplace_back(op_handle);

  auto *in =
      result->Get<GraphVars>("vars").at(src_dev_id).at(p_name).back().get();
  op_handle->AddInput(in);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>("vars").at(i).at(p_name);
    auto *out_var = new VarHandle(
        result->CreateEmptyNode(p_name, ir::Node::Type::kVariable), vars.size(),
        i, p_name, p);
    vars.emplace_back(out_var);
    op_handle->AddOutput(out_var);
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOp(Graph *result,
                                                    ir::Node *node,
                                                    int dev_id) const {
  result->Get<GraphOps>("ops").emplace_back(
      new ComputationOpHandle(result->CreateOpNode(node->Op()),
                              local_scopes_[dev_id], places_[dev_id]));
  CreateOpHandleIOs(result, node, dev_id);
}

void MultiDevSSAGraphBuilder::InsertAllReduceOp(Graph *result,
                                                const std::string &og) const {
#ifdef PADDLE_WITH_CUDA
  result->Get<GraphOps>("ops").emplace_back(new AllReduceOpHandle(
      result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>("ops").emplace_back(new AllReduceOpHandle(
      result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>("ops").back().get();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>("vars")[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());

    auto var =
        new VarHandle(result->CreateEmptyNode(og, ir::Node::Type::kVariable),
                      vars.size(), i, og, p);
    vars.emplace_back(var);
    op_handle->AddOutput(var);
  }
}

void MultiDevSSAGraphBuilder::InsertDataBalanceOp(
    Graph *result, const std::vector<std::string> &datas) const {
#ifdef PADDLE_WITH_CUDA
  result->Get<GraphOps>("ops").emplace_back(new DataBalanceOpHandle(
      result->CreateEmptyNode("data_balance", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>("ops").emplace_back(new DataBalanceOpHandle(
      result->CreateEmptyNode("data_balance", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>("ops").back().get();
  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    for (const std::string &d_name : datas) {
      auto &vars = result->Get<GraphVars>("vars")[i][d_name];
      PADDLE_ENFORCE(!vars.empty());
      op_handle->AddInput(vars.back().get());
      auto var = new VarHandle(
          result->CreateEmptyNode(d_name, ir::Node::Type::kVariable),
          vars.size(), i, d_name, p);
      vars.emplace_back(var);
      op_handle->AddOutput(var);
    }
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

int MultiDevSSAGraphBuilder::GetOpDeviceID(ir::Node *node) const {
  if (strategy_.reduce_ != BuildStrategy::ReduceStrategy::kReduce) {
    return -1;
  }
  int op_role = boost::get<int>(
      node->Op()->GetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName()));
  if (op_role != static_cast<int>(framework::OpRole::kOptimize)) {
    return -1;
  }
  auto param_grad = boost::get<std::vector<std::string>>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));

  PADDLE_ENFORCE_EQ(param_grad.size(), 2U);
  int dev_id = GetVarDeviceID(param_grad[1]);
  PADDLE_ENFORCE_NE(dev_id, -1, "dev_id should not be -1.[%s, %s]",
                    node->Op()->Type(), param_grad[0]);
  return dev_id;
}

int MultiDevSSAGraphBuilder::GetVarDeviceID(const std::string &varname) const {
  auto got = var_name_on_devices_.find(varname);
  return got == var_name_on_devices_.end() ? -1 : got->second;
}

void MultiDevSSAGraphBuilder::CreateScaleLossGradOp(Graph *result) const {
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
    auto *op_handle = new ScaleLossGradOpHandle(
        result->CreateEmptyNode("scale_loss_grad", ir::Node::Type::kOperation),
        local_scopes_.size(), local_scopes_[i], places_[i],
        communication_dev_ctx);
    result->Get<GraphOps>("ops").emplace_back(op_handle);

    // FIXME: Currently ScaleLossGradOp only use device_count as scale
    // factor. So it does not depend on any other operators.
    // VarHandle *loss = GetVarHandle(loss_var_name, place);
    // loss->pending_ops_.emplace_back(op_handle);
    // op_handle->inputs_.emplace_back(loss);

    CreateOpOutput(result, op_handle,
                   result->CreateEmptyNode(GradVarName(loss_var_name_),
                                           ir::Node::Type::kVariable),
                   places_[i], i);
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOps(Graph *result,
                                                     ir::Node *node,
                                                     size_t num_places) const {
  for (size_t scope_idx = 0; scope_idx < num_places; ++scope_idx) {
    auto p = places_[scope_idx];
    auto s = local_scopes_[scope_idx];
    result->Get<GraphOps>("ops").emplace_back(
        new ComputationOpHandle(result->CreateOpNode(node->Op()), s, p));
    CreateOpHandleIOs(result, node, scope_idx);
  }
}

VarHandle *MultiDevSSAGraphBuilder::CreateReduceOp(Graph *result,
                                                   const std::string &og,
                                                   int dst_dev_id) const {
#ifdef PADDLE_WITH_CUDA
  result->Get<GraphOps>("ops").emplace_back(new ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>("ops").emplace_back(new ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>("ops").back().get();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>("vars")[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());
  }
  auto &vars = result->Get<GraphVars>("vars")[dst_dev_id][og];
  auto var =
      new VarHandle(result->CreateEmptyNode(og, ir::Node::Type::kVariable),
                    vars.size(), dst_dev_id, og, places_[dst_dev_id]);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
  return var;
}

// Find the first occurence of `prev_op_name` and make current `op` depend
// on it.
void MultiDevSSAGraphBuilder::ConnectOp(Graph *result, OpHandleBase *op,
                                        const std::string &prev_op_name) const {
  for (auto &prev_op : result->Get<GraphOps>("ops")) {
    if (prev_op->Name() == prev_op_name) {
      auto *dep_var = new DummyVarHandle(
          result->CreateEmptyNode("dummy", ir::Node::Type::kVariable));
      prev_op->AddOutput(dep_var);
      result->Get<GraphDepVars>("dep_vars").emplace(dep_var);
      op->AddInput(dep_var);
    }
  }
}

void MultiDevSSAGraphBuilder::CreateDistTrainOp(Graph *result,
                                                ir::Node *node) const {
  int op_dev_id = -1;
  std::vector<std::string> input_var_names;
  std::vector<std::string> output_var_names;
  for (ir::Node *input : node->inputs) {
    input_var_names.push_back(input->Name());
  }
  for (ir::Node *output : node->outputs) {
    output_var_names.push_back(output->Name());
  }

  if (node->Op()->Type() == "split_byref" ||
      node->Op()->Type() == "split_selected_rows") {
    op_dev_id = GetVarDeviceID(input_var_names[0]);
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
      op_dev_id = GetAppropriateDeviceID(input_var_names);
      for (auto &varname : input_var_names) {
        var_name_on_devices_.emplace(varname, op_dev_id);
      }
    }
    for (auto &varname : output_var_names) {
      var_name_on_devices_.emplace(varname, op_dev_id);
    }
  } else if (node->Op()->Type() == "concat") {
    op_dev_id = GetVarDeviceID(input_var_names[0]);
    for (auto &varname : output_var_names) {
      var_name_on_devices_.emplace(varname, op_dev_id);
    }
  } else {
    PADDLE_ENFORCE(
        "the distribute training related op should be in [split_byref, "
        "concat].");
  }

  PADDLE_ENFORCE(op_dev_id != -1,
                 "can not find right place for distributed op: %s",
                 node->Op()->Type());

  CreateComputationalOp(result, node, op_dev_id);
  if (node->Op()->Type() == "concat") {
    ConnectOp(result, result->Get<GraphOps>("ops").back().get(),
              "fetch_barrier");
  }
}

// Create RPC related op handles that connects its in ops and out ops.
void MultiDevSSAGraphBuilder::CreateRPCOp(Graph *result, ir::Node *node) const {
  int op_dev_id = -1;
  if (node->Op()->Type() == "send") {
    op_dev_id = GetVarDeviceID(node->inputs[0]->Name());
    // the variable name which contains .block means it was splited by
    // split_byref op
    // so that we can balance the variable blocks to all the pserver
    // instances.
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce &&
        node->inputs[0]->Name().find(".block") == std::string::npos) {
      std::vector<std::string> input_var_names;
      for (ir::Node *n : node->inputs) {
        input_var_names.push_back(n->Name());
      }
      op_dev_id = GetAppropriateDeviceID(input_var_names);
      for (auto &varname : input_var_names) {
        var_name_on_devices_.emplace(varname, op_dev_id);
      }
    }
  } else if (node->Op()->Type() == "recv") {
    std::vector<std::string> output_var_names;
    for (ir::Node *n : node->outputs) {
      output_var_names.push_back(n->Name());
    }
    op_dev_id = GetAppropriateDeviceID(output_var_names);
    for (auto &varname : output_var_names) {
      var_name_on_devices_.emplace(varname, op_dev_id);
    }
  } else {
    // send_barrier and fetch_barrier op can be scheduled on device 0
    op_dev_id = 0;
  }

  PADDLE_ENFORCE(op_dev_id != -1, "can not find the right place for rpc op: %s",
                 node->Op()->Type());

  result->Get<GraphOps>("ops").emplace_back(new RPCOpHandle(
      result->CreateOpNode(node->Op()), *node->Op(), local_scopes_[op_dev_id],
      node->Op()->Type(), places_[op_dev_id]));

  if (node->Op()->Type() == "send_barrier") {
    ConnectOp(result, result->Get<GraphOps>("ops").back().get(), "send");
  } else if (node->Op()->Type() == "recv") {
    ConnectOp(result, result->Get<GraphOps>("ops").back().get(),
              "send_barrier");
  } else if (node->Op()->Type() == "fetch_barrier") {
    ConnectOp(result, result->Get<GraphOps>("ops").back().get(), "recv");
  } else if (node->Op()->Type() == "send") {
    // do nothing
  } else {
    PADDLE_THROW(
        "rpc op should be in ["
        "send, send_barrier. recv, fetch_barrier]");
  }

  CreateOpHandleIOs(result, node, op_dev_id);
}

bool MultiDevSSAGraphBuilder::IsScaleLossOp(ir::Node *node) const {
  return boost::get<int>(
             node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
             (static_cast<int>(OpRole::kBackward) |
              static_cast<int>(OpRole::kLoss)) &&
         !loss_var_name_.empty();  // If loss_var is empty. This is test mode
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
