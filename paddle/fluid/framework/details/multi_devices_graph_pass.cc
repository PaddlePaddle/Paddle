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
#include "paddle/fluid/framework/details/fused_broadcast_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_graph_pass.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/rpc_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace details {

namespace {
// TODO(panyx0718): Clean this up as well.
// all operators. NOTE that even we use a vector here, the operators is
// unordered.
typedef std::vector<OpHandleBase *> GraphOps;
const char kGraphOps[] = "ops";

bool IsSameOpRole(const ir::Node &node, const framework::OpRole &role) {
  return boost::get<int>(
             node.Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
         static_cast<int>(role);
}

void PolishGraphToSupportDataHazards(ir::Graph *graph) {
  for (auto &var_map : graph->Get<GraphVars>(kGraphVars)) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        continue;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        OpHandleBase *write_op = (*it_new)->GeneratedOp();
        const auto &read_ops = (*it_old)->PendingOps();

        for (auto *read_op : read_ops) {
          // Manually add a dependency var from read_op to write_op;
          if (read_op == write_op) {
            // Read Write is the same op.
            continue;
          }
          bool has_dep = false;
          for (auto *r_out : read_op->Outputs()) {
            for (auto *w_in : write_op->Inputs()) {
              if (r_out->Node() == w_in->Node()) {
                has_dep = true;
                break;
              }
            }
          }
          if (has_dep) continue;

          auto *dep_var = new DummyVarHandle(graph->CreateControlDepVar());
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);
        }
      }
    }
  }
}

VarHandle *CreateOrGetLatestVarHandle(ir::Graph *graph, ir::Node *node,
                                      const platform::Place &place,
                                      size_t place_offset) {
  auto &var_holders = graph->Get<GraphVars>(kGraphVars)[place_offset];
  auto &var_holder = var_holders[node->Name()];
  VarHandle *var = nullptr;
  if (var_holder.empty()) {
    if (node->Var()) {
      var = new VarHandle(graph->CreateVarNode(node->Var()), 0, place_offset,
                          node->Name(), place);
    } else {
      var = new VarHandle(
          graph->CreateEmptyNode(node->Name(), ir::Node::Type::kVariable), 0,
          place_offset, node->Name(), place);
    }
    var_holder.emplace_back(var);
  } else {
    var = *var_holder.rbegin();
  }
  return var;
}

void CreateOpOutput(ir::Graph *graph, OpHandleBase *op_handle,
                    ir::Node *new_node, const platform::Place &place,
                    size_t place_offset) {
  auto &vars =
      graph->Get<GraphVars>(kGraphVars)[place_offset][new_node->Name()];
  size_t version = vars.size();
  auto var =
      new VarHandle(new_node, version, place_offset, new_node->Name(), place);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
}

void AddOutputToLeafOps(ir::Graph *graph) {
  for (auto &op : graph->Get<GraphOps>(kGraphOps)) {
    if (!op->Outputs().empty()) {
      continue;
    }
    auto *dummy_leaf = new DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}
}  // namespace

static const char kLossVarName[] = "loss_var_name";
static const char kPlaces[] = "places";
static const char kLocalScopes[] = "local_scopes";
static const char kStrategy[] = "strategy";
static const char kNumTrainers[] = "num_trainers";

void MultiDevSSAGraphBuilder::Init() const {
  all_vars_.clear();
  balance_vars_.clear();

  loss_var_name_ = Get<const std::string>(kLossVarName);
  places_ = Get<const std::vector<platform::Place>>(kPlaces);
  local_scopes_ = Get<const std::vector<Scope *>>(kLocalScopes);
  strategy_ = Get<const BuildStrategy>(kStrategy);
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  nccl_ctxs_ = &Get<platform::NCCLContextMap>("nccl_ctxs");
#endif

  balance_vars_.resize(places_.size(), 0);

  if (strategy_.enable_data_balance_ && places_.size() == 1) {
    LOG(WARNING) << "It is no need to enable data balance when there is only "
                    "one place. enable_data_balance is set to False.";
    strategy_.enable_data_balance_ = false;
  }
}

std::unique_ptr<ir::Graph> MultiDevSSAGraphBuilder::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  Init();
  // Give the topology sort order and rebuild the graph structure.
  std::vector<ir::Node *> sorted_ops = ir::TopologySortOperations(*graph);

  if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kReduce) {
    sorted_ops = SortForReduce(sorted_ops);
  }

  auto nodes = graph->ReleaseNodes();
  ir::Graph &result = *graph;

  int num_trainers = Get<int>(kNumTrainers);

  for (auto &node : nodes) {
    if (node->IsVar() && node->Var()) {
      all_vars_.emplace(node->Name(), node->Var());
    }
  }

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.Set(kGraphVars, new GraphVars(places_.size()));
  result.Set(kGraphDepVars, new GraphDepVars);
  result.Set(kGraphOps, new GraphOps);

  std::vector<std::unordered_set<std::string>> bcast_var_name_set;
  bcast_var_name_set.resize(places_.size());

  bool is_forwarding = true;
  bool is_dist_train = false;

  std::unordered_map<std::string, int> shared_var_device;

  for (ir::Node *node : sorted_ops) {
    if (IsSameOpRole(*node, OpRole::kRPC)) {
      int op_dev_id = CreateRPCOp(&result, node, &shared_var_device);
      PADDLE_ENFORCE(op_dev_id != -1,
                     "Can not schedule the RPC operator to the right place.");
      if (node->Op()->Type() == "recv") {
        auto recv_vars_attr =
            boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
                OpProtoAndCheckerMaker::OpRoleVarAttrName()));
        PADDLE_ENFORCE(recv_vars_attr.size() == 2UL);  // [parameter, gradient]
        if (recv_vars_attr[0].find(".block") == std::string::npos) {
          bcast_var_name_set[op_dev_id].emplace(recv_vars_attr[0]);
        }
      }
      is_dist_train = true;
    } else if (IsSameOpRole(*node, OpRole::kDist)) {
      int op_dev_id = CreateDistTrainOp(&result, node, &shared_var_device);
      if (node->Op()->Type() == "concat") {
        auto origin_param_name = node->Op()->OutputArgumentNames()[0];
        bcast_var_name_set[op_dev_id].emplace(origin_param_name);
      }
    } else if (IsScaleLossOp(node)) {
      // user can customize loss@grad if not use_default_grad_scale_
      if (strategy_.gradient_scale_ !=
          BuildStrategy::GradientScaleStrategy::kCustomized) {
        // TODO(paddle-dev): Why is there no input for this op_handle?
        auto loss_grad_name = node->Op()->OutputArgumentNames()[0];
        CreateScaleLossGradOp(&result, loss_grad_name, node->outputs[0]);
      }
      // This assumes the backward generating code will ensure IsScaleLossOp
      // is true only for the op that scale the final scalar loss.
      // It also assumes backward op will always follow the forward op in
      // the block.
      is_forwarding = false;
    } else {
      int op_dev_id = GetOpDeviceID(node, shared_var_device);
      if (op_dev_id != -1) {  // This op only runs on one specific device.
        CreateComputationalOp(&result, node, op_dev_id);
        for (ir::Node *n : node->outputs) {
          shared_var_device.emplace(n->Name(), op_dev_id);
        }
      } else {
        // This op runs on all devices, and its output may have parameter's
        // gradients.
        // TODO(paddle-dev): Why is so special about "read" op?
        if (node->Op()->Type() == "read" && strategy_.enable_data_balance_) {
          node->Op()->SetAttr("throw_eof_exp", false);
          CreateComputationalOps(&result, node, places_.size());
          const auto &data_var_names = node->Op()->Output("Out");
          InsertDataBalanceOp(&result, data_var_names);
        } else {
          CreateComputationalOps(&result, node, places_.size());
        }

        if (!is_forwarding && (places_.size() > 1 || num_trainers > 1)) {
          bool is_bk_op =
              static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                    OpProtoAndCheckerMaker::OpRoleAttrName())) &
                                static_cast<int>(OpRole::kBackward));
          if (!is_bk_op) continue;
          // Currently, we assume that once gradient is generated, it can be
          // broadcast, and each gradient is only broadcast once.
          try {
            auto backward_vars = boost::get<std::vector<std::string>>(
                node->Op()->GetNullableAttr(
                    OpProtoAndCheckerMaker::OpRoleVarAttrName()));

            PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);

            for (size_t i = 0; i < backward_vars.size(); i += 2) {
              auto &p_name = backward_vars[i];
              auto &g_name = backward_vars[i + 1];
              VLOG(10) << "Bcast " << g_name << " for parameter " << p_name;
              size_t cur_device_id = -1;
              switch (strategy_.reduce_) {
                case BuildStrategy::ReduceStrategy::kReduce:
                  cur_device_id = GetAppropriateDeviceID({g_name});
                  CreateReduceOp(&result, g_name, cur_device_id);
                  shared_var_device.emplace(g_name, cur_device_id);
                  if (!is_dist_train) {
                    bcast_var_name_set[cur_device_id].emplace(p_name);
                  }
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
  bool use_gpu = false;
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  use_gpu = nccl_ctxs_ != nullptr;
#endif

  // Insert broadcast operators principle:
  // 1. Broadcast optimized parameters in Reduce strategy;
  // 2. No need broadcast optimized parameters in AllReduce strategy because of
  //    the optimization sub-graph would be run on every GPU;
  // 3. Allways broadcast received parameters in Distribute Training.
  if ((use_gpu &&
       strategy_.reduce_ == BuildStrategy::ReduceStrategy::kReduce) ||
      is_dist_train) {
    if (strategy_.fuse_broadcast_op_) {
      CreateFusedBroadcastOp(&result, bcast_var_name_set);
    } else {
      for (size_t dev_id = 0; dev_id < bcast_var_name_set.size(); ++dev_id) {
        auto &to_bcast_set = bcast_var_name_set[dev_id];
        for (auto &bcast_name : to_bcast_set) {
          CreateBroadcastOp(&result, bcast_name, dev_id);
        }
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
  result.Erase<GraphOps>(kGraphOps);
  return graph;
}

std::vector<ir::Node *> MultiDevSSAGraphBuilder::SortForReduce(
    const std::vector<ir::Node *> &topo_ops) const {
  std::unordered_map<std::string, int> shared_var_device;
  std::vector<ir::Node *> sorted_ops;
  std::unordered_map<std::string, std::vector<ir::Node *>> delayed_op;

  auto insert_delayed_op = [&](const std::string &var_name, int dev_id) {
    shared_var_device.emplace(var_name, dev_id);
    if (delayed_op.count(var_name)) {
      auto &ops = delayed_op.at(var_name);
      sorted_ops.insert(sorted_ops.end(), ops.begin(), ops.end());
      delayed_op.at(var_name).clear();
    }
  };

  for (ir::Node *node : topo_ops) {
    int op_dev_id = GetOpDeviceID(node, shared_var_device, &delayed_op);
    if (op_dev_id > -1) {
      // This op only runs on one specific device.
      sorted_ops.emplace_back(node);
      for (ir::Node *n : node->outputs) {
        insert_delayed_op(n->Name(), op_dev_id);
      }
    } else if (op_dev_id == -1) {
      // This op runs on all devices, and its output may have parameter's
      // gradients.
      sorted_ops.emplace_back(node);
      bool is_bk_op =
          static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                OpProtoAndCheckerMaker::OpRoleAttrName())) &
                            static_cast<int>(OpRole::kBackward));
      if (!is_bk_op) continue;
      // Currently, we assume that once gradient is generated, it can be
      // broadcast, and each gradient is only broadcast once.
      std::vector<std::string> backward_vars;
      try {
        backward_vars =
            boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
                OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      } catch (boost::bad_get e) {
      }
      PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);

      for (size_t i = 0; i < backward_vars.size(); i += 2) {
        auto &g_name = backward_vars[i + 1];
        size_t cur_device_id = GetAppropriateDeviceID({g_name});
        insert_delayed_op(g_name, static_cast<int>(cur_device_id));
      }
    } else if (op_dev_id == -2) {
      // The Op on which the Op depends has not yet been generated.
    }
  }

  PADDLE_ENFORCE_EQ(sorted_ops.size(), topo_ops.size());
  return sorted_ops;
}

void MultiDevSSAGraphBuilder::CreateOpHandleIOs(ir::Graph *result,
                                                ir::Node *node,
                                                size_t place_id) const {
  auto p = places_[place_id];
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
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

size_t MultiDevSSAGraphBuilder::GetAppropriateDeviceID(
    const std::vector<std::string> &var_names) const {
  int64_t numel_sum = 0;
  for (auto var_name : var_names) {
    if (all_vars_.find(var_name) == all_vars_.end()) continue;
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

void MultiDevSSAGraphBuilder::SetCommunicationContext(
    OpHandleBase *op_handle, const platform::Place &p) const {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  if (nccl_ctxs_ == nullptr) {
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#else
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));
#endif
}

void MultiDevSSAGraphBuilder::CreateBroadcastOp(ir::Graph *result,
                                                const std::string &p_name,
                                                size_t src_dev_id) const {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  auto *op_handle = new BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_);
#else
  auto *op_handle = new BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_);
#endif
  result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

  auto *in =
      result->Get<GraphVars>(kGraphVars).at(src_dev_id).at(p_name).back();
  op_handle->AddInput(in);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>(kGraphVars).at(i).at(p_name);
    auto *out_var = new VarHandle(
        result->CreateEmptyNode(p_name, ir::Node::Type::kVariable), vars.size(),
        i, p_name, p);
    vars.emplace_back(out_var);
    op_handle->AddOutput(out_var);
  }
}

void MultiDevSSAGraphBuilder::CreateFusedBroadcastOp(
    ir::Graph *result,
    const std::vector<std::unordered_set<std::string>> &bcast_varnames) const {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  auto *op_handle = new FusedBroadcastOpHandle(
      result->CreateEmptyNode("fused_broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_);
#else
  auto *op_handle = new FusedBroadcastOpHandle(
      result->CreateEmptyNode("fused_broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_);
#endif
  result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
  }

  for (size_t dev_id = 0; dev_id < bcast_varnames.size(); ++dev_id) {
    for (auto &p_name : bcast_varnames[dev_id]) {
      auto *in =
          result->Get<GraphVars>(kGraphVars).at(dev_id).at(p_name).back();
      op_handle->AddInput(in);
      for (size_t out_dev_id = 0; out_dev_id < places_.size(); ++out_dev_id) {
        auto &p = places_[out_dev_id];
        auto &vars =
            result->Get<GraphVars>(kGraphVars).at(out_dev_id).at(p_name);
        auto *out_var = new VarHandle(
            result->CreateEmptyNode(p_name, ir::Node::Type::kVariable),
            vars.size(), out_dev_id, p_name, p);
        vars.emplace_back(out_var);
        op_handle->AddOutput(out_var);
      }
    }
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOp(ir::Graph *result,
                                                    ir::Node *node,
                                                    int dev_id) const {
  result->Get<GraphOps>(kGraphOps).emplace_back(
      new ComputationOpHandle(result->CreateOpNode(node->Op()),
                              local_scopes_[dev_id], places_[dev_id], dev_id));
  CreateOpHandleIOs(result, node, dev_id);
}

void MultiDevSSAGraphBuilder::InsertAllReduceOp(ir::Graph *result,
                                                const std::string &og) const {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  result->Get<GraphOps>(kGraphOps).emplace_back(new AllReduceOpHandle(
      result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>(kGraphOps).emplace_back(new AllReduceOpHandle(
      result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>(kGraphVars)[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad);

    auto var =
        new VarHandle(result->CreateEmptyNode(og, ir::Node::Type::kVariable),
                      vars.size(), i, og, p);
    vars.emplace_back(var);
    op_handle->AddOutput(var);
  }
}

void MultiDevSSAGraphBuilder::InsertDataBalanceOp(
    ir::Graph *result, const std::vector<std::string> &datas) const {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  result->Get<GraphOps>(kGraphOps).emplace_back(new DataBalanceOpHandle(
      result->CreateEmptyNode("data_balance", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>(kGraphOps).emplace_back(new DataBalanceOpHandle(
      result->CreateEmptyNode("data_balance", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    for (const std::string &d_name : datas) {
      auto &vars = result->Get<GraphVars>(kGraphVars)[i][d_name];
      PADDLE_ENFORCE(!vars.empty());
      op_handle->AddInput(vars.back());
      auto var = new VarHandle(
          result->CreateEmptyNode(d_name, ir::Node::Type::kVariable),
          vars.size(), i, d_name, p);
      vars.emplace_back(var);
      op_handle->AddOutput(var);
    }
  }
}

int MultiDevSSAGraphBuilder::GetOpDeviceID(
    ir::Node *node,
    const std::unordered_map<std::string, int> &shared_var_device,
    std::unordered_map<std::string, std::vector<ir::Node *>> *delay_ops) const {
  if (strategy_.reduce_ != BuildStrategy::ReduceStrategy::kReduce) {
    return -1;
  }

  if (!IsSameOpRole(*node, framework::OpRole::kOptimize)) {
    return -1;
  }

  auto param_grad = boost::get<std::vector<std::string>>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));

  PADDLE_ENFORCE_EQ(param_grad.size(), 2U);
  int dev_id = GetVarDeviceID(param_grad[1], shared_var_device);

  if (dev_id == -1) {
    (*delay_ops)[param_grad[1]].push_back(node);
    return -2;
  }
  return dev_id;
}

int MultiDevSSAGraphBuilder::GetOpDeviceID(
    ir::Node *node,
    const std::unordered_map<std::string, int> &shared_var_device) const {
  if (strategy_.reduce_ != BuildStrategy::ReduceStrategy::kReduce) {
    return -1;
  }

  if (!IsSameOpRole(*node, framework::OpRole::kOptimize)) {
    return -1;
  }
  auto param_grad = boost::get<std::vector<std::string>>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));

  PADDLE_ENFORCE_EQ(param_grad.size(), 2U);
  int dev_id = GetVarDeviceID(param_grad[1], shared_var_device);
  PADDLE_ENFORCE_NE(dev_id, -1, "dev_id should not be -1.[%s, %s, %s]",
                    node->Op()->Type(), param_grad[0], param_grad[1]);
  return dev_id;
}

int MultiDevSSAGraphBuilder::GetVarDeviceID(
    const std::string &varname,
    const std::unordered_map<std::string, int> &shared_var_device) const {
  auto got = shared_var_device.find(varname);
  if (got == shared_var_device.end()) {
    auto pos = varname.find(framework::kNewGradSuffix);
    if (pos != std::string::npos) {
      got = shared_var_device.find(varname.substr(0, pos));
    }
  }
  return got == shared_var_device.end() ? -1 : got->second;
}

void MultiDevSSAGraphBuilder::CreateScaleLossGradOp(
    ir::Graph *result, const std::string &loss_grad_name,
    ir::Node *out_var_node) const {
  for (size_t i = 0; i < places_.size(); ++i) {
    // Insert ScaleCost OpHandle
    auto *dev_ctx = platform::DeviceContextPool::Instance().Get(places_[i]);
    auto *op_handle = new ScaleLossGradOpHandle(
        result->CreateEmptyNode("scale_loss_grad", ir::Node::Type::kOperation),
        local_scopes_.size(), local_scopes_[i], places_[i], dev_ctx);
    result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

    // FIXME: Currently ScaleLossGradOp only use device_count as scale
    // factor. So it does not depend on any other operators.
    // VarHandle *loss = GetVarHandle(loss_var_name, place);
    // loss->pending_ops_.emplace_back(op_handle);
    // op_handle->inputs_.emplace_back(loss);

    CreateOpOutput(result, op_handle,
                   result->CreateVarNode(out_var_node->Var()), places_[i], i);
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOps(ir::Graph *result,
                                                     ir::Node *node,
                                                     size_t num_places) const {
  for (size_t scope_idx = 0; scope_idx < num_places; ++scope_idx) {
    auto p = places_[scope_idx];
    auto s = local_scopes_[scope_idx];
    result->Get<GraphOps>(kGraphOps).emplace_back(new ComputationOpHandle(
        result->CreateOpNode(node->Op()), s, p, scope_idx));
    CreateOpHandleIOs(result, node, scope_idx);
  }
}

VarHandle *MultiDevSSAGraphBuilder::CreateReduceOp(ir::Graph *result,
                                                   const std::string &og,
                                                   int dst_dev_id) const {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  result->Get<GraphOps>(kGraphOps).emplace_back(new ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>(kGraphOps).emplace_back(new ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>(kGraphVars)[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad);
  }
  auto &vars = result->Get<GraphVars>(kGraphVars)[dst_dev_id][og];
  auto var =
      new VarHandle(result->CreateEmptyNode(og, ir::Node::Type::kVariable),
                    vars.size(), dst_dev_id, og, places_[dst_dev_id]);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
  return var;
}

int MultiDevSSAGraphBuilder::CreateDistTrainOp(
    ir::Graph *result, ir::Node *node,
    std::unordered_map<std::string, int> *shared_var_device) const {
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
      node->Op()->Type() == "split_selected_rows" ||
      node->Op()->Type() == "split_ids") {
    // TODO(paddle-dev): getting the first var is not safe.
    op_dev_id = GetVarDeviceID(input_var_names[0], *shared_var_device);
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
      op_dev_id = GetAppropriateDeviceID(input_var_names);
      for (auto &varname : input_var_names) {
        shared_var_device->emplace(varname, op_dev_id);
      }
    }
    for (auto &varname : output_var_names) {
      shared_var_device->emplace(varname, op_dev_id);
    }
  } else if (node->Op()->Type() == "concat") {
    op_dev_id = GetVarDeviceID(input_var_names[0], *shared_var_device);
    for (auto &varname : output_var_names) {
      shared_var_device->emplace(varname, op_dev_id);
    }
  } else {
    LOG(ERROR) << "got unexpected dist op: " << node->Op()->Type();
    PADDLE_THROW(
        "the distribute training related op should be in [split_byref, "
        "concat].");
  }

  PADDLE_ENFORCE(op_dev_id != -1,
                 "can not find right place for distributed op: %s",
                 node->Op()->Type());

  CreateComputationalOp(result, node, op_dev_id);
  return op_dev_id;
}

void SetOpInputsAllPlaces(ir::Graph *result, ir::Node *node, int num_places) {
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
  for (ir::Node *input : node->inputs) {
    VarHandle *var = nullptr;
    for (int place_offset = 0; place_offset < num_places; ++place_offset) {
      auto &var_holders = result->Get<GraphVars>(kGraphVars)[place_offset];
      auto &var_holder = var_holders[input->Name()];
      if (!var_holder.empty()) {
        var = *var_holder.rbegin();
        op_handle->AddInput(var);
      }
    }
  }
}

// Create RPC related op handles that connects its in ops and out ops.
int MultiDevSSAGraphBuilder::CreateRPCOp(
    ir::Graph *result, ir::Node *node,
    std::unordered_map<std::string, int> *shared_var_device) const {
  int op_dev_id = -1;
  if (node->Op()->Type() == "send") {
    // TODO(paddle-dev): getting the first var is not safe.
    op_dev_id = GetVarDeviceID(node->inputs[0]->Name(), *shared_var_device);
    PADDLE_ENFORCE(!ir::IsControlDepVar(*node->inputs[0]),
                   "This hack no longer holds, please fix.");
    // the variable name which contains .block means it was splited by
    // split_byref op
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce &&
        node->inputs[0]->Name().find(".block") == std::string::npos) {
      std::vector<std::string> input_var_names;
      for (ir::Node *n : node->inputs) {
        input_var_names.push_back(n->Name());
      }
      auto send_param_grad = boost::get<std::vector<std::string>>(
          node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      PADDLE_ENFORCE_EQ(send_param_grad.size(), 2U);
      op_dev_id = GetAppropriateDeviceID({send_param_grad[1]});
      VLOG(10) << "send grad " << input_var_names[0] << " origin "
               << send_param_grad[1] << " place: " << op_dev_id;
      for (auto &varname : input_var_names) {
        shared_var_device->emplace(varname, op_dev_id);
      }
      shared_var_device->emplace(send_param_grad[1], op_dev_id);
    }
  } else if (node->Op()->Type() == "recv") {
    std::vector<std::string> output_var_names;
    for (ir::Node *n : node->outputs) {
      output_var_names.push_back(n->Name());
    }
    auto recv_param_grad = boost::get<std::vector<std::string>>(
        node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));
    if (recv_param_grad.size() == 2U) {
      op_dev_id = GetVarDeviceID(recv_param_grad[1], *shared_var_device);
      VLOG(10) << "recv param " << recv_param_grad[0]
               << " get grad place: " << recv_param_grad[1]
               << " place: " << op_dev_id;
    } else {
      op_dev_id = GetAppropriateDeviceID(output_var_names);
    }
    for (auto &varname : output_var_names) {
      shared_var_device->emplace(varname, op_dev_id);
    }
  } else {
    // send_barrier, fetch_barrier will run on place 0;
    op_dev_id = 0;
  }

  PADDLE_ENFORCE(op_dev_id != -1, "can not find the right place for rpc op: %s",
                 node->Op()->Type());
  result->Get<GraphOps>(kGraphOps).emplace_back(new RPCOpHandle(
      result->CreateOpNode(node->Op()), *node->Op(), local_scopes_[op_dev_id],
      node->Op()->Type(), places_[op_dev_id]));

  if (node->Op()->Type() == "send") {
    CreateOpHandleIOs(result, node, op_dev_id);
  } else {
    // send_barrier, recv, fetch_barrier's inputs are deps var, get them from
    // all places
    auto p = places_[op_dev_id];
    auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));

    SetOpInputsAllPlaces(result, node, places_.size());
    for (ir::Node *output : node->outputs) {
      int outvar_dev_id = op_dev_id;
      if (node->Op()->Type() == "fetch_barrier") {
        outvar_dev_id = GetVarDeviceID(output->Name(), *shared_var_device);
        PADDLE_ENFORCE_NE(outvar_dev_id, -1, "output name %s", output->Name());
      }
      p = places_[outvar_dev_id];
      ir::Node *new_node = nullptr;
      if (output->Var()) {
        new_node = result->CreateVarNode(output->Var());
      } else {
        new_node =
            result->CreateEmptyNode(output->Name(), ir::Node::Type::kVariable);
      }
      CreateOpOutput(result, op_handle, new_node, p, outvar_dev_id);
    }
  }
  return op_dev_id;
}

bool MultiDevSSAGraphBuilder::IsSparseGradient(const std::string &og) const {
  PADDLE_ENFORCE(all_vars_.count(og) != 0);
  if (all_vars_.at(og)->GetType() == proto::VarType::SELECTED_ROWS) {
    return true;
  }
  return false;
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

REGISTER_PASS(multi_devices_pass,
              paddle::framework::details::MultiDevSSAGraphBuilder)
    .RequirePassAttr(paddle::framework::details::kLossVarName)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kLocalScopes)
    .RequirePassAttr(paddle::framework::details::kStrategy)
    .RequirePassAttr(paddle::framework::details::kNumTrainers);
