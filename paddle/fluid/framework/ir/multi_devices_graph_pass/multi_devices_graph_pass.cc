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
#include "paddle/fluid/framework/ir/multi_devices_graph_pass/multi_devices_graph_pass.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle::framework::ir {

namespace {
// TODO(panyx0718): Clean this up as well.
// all operators. NOTE that even we use a vector here, the operators is
// unordered.
typedef std::vector<details::OpHandleBase *> GraphOps;
const char kGraphOps[] = "ops";  // NOLINT

bool OpHaveRole(const ir::Node &node, const framework::OpRole &role) {
  return PADDLE_GET_CONST(
             int,
             node.Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
         static_cast<int>(role);
}

void PolishGraphToSupportDataHazards(ir::Graph *graph) {
  for (auto &var_map : graph->Get<details::GraphVars>(details::kGraphVars)) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        continue;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        details::OpHandleBase *write_op = (*it_new)->GeneratedOp();
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

          auto *dep_var =
              new details::DummyVarHandle(graph->CreateControlDepVar());
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->Get<details::GraphDepVars>(details::kGraphDepVars)
              .emplace(dep_var);
        }
      }
    }
  }
}

details::VarHandle *CreateOrGetLatestVarHandle(ir::Graph *graph,
                                               ir::Node *node,
                                               const phi::Place &place,
                                               size_t place_offset) {
  auto &var_holders =
      graph->Get<details::GraphVars>(details::kGraphVars)[place_offset];
  auto &var_holder = var_holders[node->Name()];
  details::VarHandle *var = nullptr;
  if (var_holder.empty()) {
    if (node->Var()) {
      var = new details::VarHandle(
          graph->CreateVarNode(node->Var(), node->GetVarNodeBlockId()),
          0,
          place_offset,
          node->Name(),
          place);
    } else {
      var = new details::VarHandle(
          graph->CreateEmptyNode(node->Name(), ir::Node::Type::kVariable),
          0,
          place_offset,
          node->Name(),
          place);
    }
    var_holder.emplace_back(var);
  } else {
    var = *var_holder.rbegin();
  }
  return var;
}

void CreateOpOutput(ir::Graph *graph,
                    details::OpHandleBase *op_handle,
                    ir::Node *new_node,
                    const phi::Place &place,
                    size_t place_offset) {
  auto &vars = graph->Get<details::GraphVars>(
      details::kGraphVars)[place_offset][new_node->Name()];
  size_t version = vars.size();
  auto var = new details::VarHandle(
      new_node, version, place_offset, new_node->Name(), place);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
}

void AddOutputToLeafOps(ir::Graph *graph) {
  for (auto &op : graph->Get<GraphOps>(kGraphOps)) {
    if (!op->Outputs().empty()) {
      continue;
    }
    auto *dummy_leaf =
        new details::DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<details::GraphDepVars>(details::kGraphDepVars)
        .emplace(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}
}  // namespace

void MultiDevSSAGraphBuilderBase::CheckGraph(const ir::Graph &graph) const {}

void MultiDevSSAGraphBuilderBase::Init() const {
  all_vars_.clear();

  loss_var_name_ = Get<const std::string>(kLossVarName);
  VLOG(10) << "Init MultiDevSSAGraphBuilder, loss name: " << loss_var_name_;
  places_ = Get<const std::vector<phi::Place>>(details::kPlaces);
  local_scopes_ = Get<const std::vector<Scope *>>(details::kLocalScopes);
  strategy_ = Get<const details::BuildStrategy>(kStrategy);
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  multi_nccl_ctxs_ = &Get<platform::NCCLCommunicator>(details::kNCCLCtxs);
  nccl_ctxs_ = nullptr;
  if (multi_nccl_ctxs_) {
    nccl_ctxs_ = multi_nccl_ctxs_->DefaultFlatCtx();
  }
#elif defined(PADDLE_WITH_XPU_BKCL)
  multi_bkcl_ctxs_ = &Get<platform::BKCLCommunicator>(details::kBKCLCtxs);
  bkcl_ctxs_ = nullptr;
  if (multi_bkcl_ctxs_) {
    bkcl_ctxs_ = multi_bkcl_ctxs_->DefaultFlatCtx();
  }
#endif
  PADDLE_ENFORCE_EQ(
      places_.size(),
      local_scopes_.size(),
      platform::errors::InvalidArgument(
          "Places size and LocalScopes not equal "
          "Places size(%d), LocalScopes size(%d) "
          "If use multi devices, Places size must equas to LocalScopes size.",
          places_.size(),
          local_scopes_.size()));
}

void MultiDevSSAGraphBuilderBase::ApplyImpl(ir::Graph *graph) const {
  Init();
  CheckGraph(*graph);
  std::vector<ir::Node *> sorted_ops = SortOperations(*graph);

  auto nodes = graph->ReleaseNodes();
  ir::Graph &result = *graph;

  std::vector<ir::Node *> isolated_vars;

  for (auto &node : nodes) {
    if (node->IsVar() && node->Var()) {
      all_vars_.emplace(node->Name(), node->Var());

      if (node->inputs.empty() && node->outputs.empty()) {
        isolated_vars.emplace_back(node.get());
      }
    }
  }

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.Set(details::kGraphVars, new details::GraphVars(places_.size()));
  result.Set(details::kGraphDepVars, new details::GraphDepVars);
  result.Set(kGraphOps, new GraphOps);

  for (auto *var_node : isolated_vars) {
    CreateIsolatedVarNode(&result, var_node);
  }

  // bool is_forwarding = true;

  for (ir::Node *node : sorted_ops) {
    if (DealWithSpecialOp(&result, node)) {
      continue;
    } else {
      // This op runs on all devices
      if (IsScaleLossOp(node)) {
        // user can customize loss@grad if not use_default_grad_scale_
        InsertScaleLossGradOp(&result, node);
        // This assumes the backward generating code will ensure IsScaleLossOp
        // is true only for the op that scale the final scalar loss.
        // It also assumes backward op will always follow the forward op in
        // the block.
        // is_forwarding = false;
      } else {
        CreateComputationalOps(&result, node, places_.size());
      }

      // // Insert collective ops if nranks > 1
      // if (!is_forwarding && Get<size_t>(details::kNRanks) > 1) {
      //   auto &op_desc = *(node->Op());
      //   bool is_bk_op = details::IsOpRole(op_desc, OpRole::kBackward);
      //   // optimize op is already processed in DealWithSpecialOp,
      //   // here we only consider backward op
      //   if (!is_bk_op) continue;

      //   /*
      //    * the op that will generate the gradient of on parameter will have
      //    one attr op_role_var
      //    * to record the parameter and gradient, like:
      //     attrs {
      //       name: "op_role_var"
      //       type: STRINGS
      //       strings: "fc_1.b_0"
      //       strings: "fc_1.b_0@GRAD"
      //     }
      //    */

      //   // Currently, we assume that once gradient is generated, it can be
      //   // broadcast, and each gradient is only broadcast once.
      //   auto backward_vars = details::GetOpRoleVarsOrEmpty(op_desc);
      //   for (size_t i = 0; i < backward_vars.size(); i += 2) {
      //     auto &p_name = backward_vars[i];
      //     auto &g_name = backward_vars[i + 1];
      //     VLOG(10) << "Bcast " << g_name << " for parameter " << p_name
      //              << " op_type " << node->Op()->Type();
      //     if (NeedCollectiveForGrad(g_name, sorted_ops)) {
      //       // InsertCollectiveOp(&result, node, p_name, g_name);
      //     }
      //   }
      // }
    }
  }

  // InsertPostprocessOps(&result);

  /*
  Dependency graph has been constructed. However, there are still data
  hazards need to be handled.
  */
  // PolishGraphToSupportDataHazards(&result);

  /*
   * Only variables should be the leaves of graph.
   */
  // AddOutputToLeafOps(&result);

  // result.Erase(kGraphOps);
}

void MultiDevSSAGraphBuilderBase::InsertScaleLossGradOp(
    ir::Graph *result, const ir::Node *node) const {
  // user can customize loss@grad if not use_default_grad_scale_
  size_t loss_scale = 0;
  switch (this->strategy_.gradient_scale_) {
    case details::BuildStrategy::GradientScaleStrategy::kOne:
      loss_scale = 1;
      break;
    case details::BuildStrategy::GradientScaleStrategy::kCoeffNumDevice:
      loss_scale = Get<size_t>(details::kNRanks);
      break;
    case details::BuildStrategy::GradientScaleStrategy::kCustomized:
      loss_scale = 0;
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unknown gradient scale strategy. Now only supports One, "
          "CoeffNumDevice and Customized strategies."));
      break;
  }

  VLOG(3) << "loss_scale: " << loss_scale;

  if (loss_scale) {
    // TODO(paddle-dev): Why is there no input for this op_handle?
    auto loss_grad_name = node->Op()->OutputArgumentNames()[0];
    auto out_dtype = this->all_vars_.at(loss_grad_name)->GetDataType();
    this->CreateScaleLossGradOp(
        result, loss_grad_name, node->outputs[0], loss_scale, out_dtype);
  }
}

bool MultiDevSSAGraphBuilderBase::DealWithSpecialOp(ir::Graph *result,
                                                    ir::Node *node) const {
  return false;
}

std::vector<ir::Node *> MultiDevSSAGraphBuilderBase::SortOperations(
    const ir::Graph &graph) const {
  return ir::TopologySortOperations(graph);
}

bool MultiDevSSAGraphBuilderBase::UseGPU() const {
  bool use_gpu = false;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  use_gpu = nccl_ctxs_ != nullptr;
#endif
  return use_gpu;
}

bool MultiDevSSAGraphBuilderBase::NeedCollectiveForGrad(
    const std::string &grad_name, std::vector<ir::Node *> ops) const {
  // if we have allreduce_op for current gradient variable in the graph,
  // then we don't need to add allreduce_op_handle for this gradient
  // NOTE: This is for the case that all gradients should add collective ops
  for (auto *node : ops) {
    if (node->Op()->Type() != "allreduce") continue;
    for (auto const &in_name : node->Op()->InputArgumentNames()) {
      if (in_name == grad_name) {
        return false;
      }
    }
  }
  return true;
}

void MultiDevSSAGraphBuilderBase::CreateOpHandleIOs(ir::Graph *result,
                                                    ir::Node *node,
                                                    size_t place_id) const {
  auto p = places_[place_id];
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));

  for (ir::Node *input : node->inputs) {
    details::VarHandle *var =
        CreateOrGetLatestVarHandle(result, input, p, place_id);
    op_handle->AddInput(var);
  }

  for (ir::Node *output : node->outputs) {
    ir::Node *new_node = nullptr;
    if (output->Var()) {
      new_node =
          result->CreateVarNode(output->Var(), output->GetVarNodeBlockId());
    } else {
      new_node =
          result->CreateEmptyNode(output->Name(), ir::Node::Type::kVariable);
    }
    CreateOpOutput(result, op_handle, new_node, p, place_id);
  }
}

void MultiDevSSAGraphBuilderBase::SetCommunicationContext(
    details::OpHandleBase *op_handle, const phi::Place &p) const {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  if (nccl_ctxs_ == nullptr) {
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#elif defined(PADDLE_WITH_XPU_BKCL)
  if (bkcl_ctxs_ == nullptr) {
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#else
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));
#endif
}

void MultiDevSSAGraphBuilderBase::CreateComputationalOp(ir::Graph *result,
                                                        ir::Node *node,
                                                        size_t dev_id) const {
  result->Get<GraphOps>(kGraphOps).emplace_back(
      new details::ComputationOpHandle(result->CreateOpNode(node->Op()),
                                       local_scopes_[dev_id],
                                       places_[dev_id],
                                       dev_id));
  CreateOpHandleIOs(result, node, dev_id);
}

void MultiDevSSAGraphBuilderBase::CreateScaleLossGradOp(
    ir::Graph *result,
    const std::string &loss_grad_name,
    ir::Node *out_var_node,
    size_t loss_scale,
    proto::VarType::Type dtype) const {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto *dev_ctx = platform::DeviceContextPool::Instance().Get(places_[i]);
    auto *op_handle = new details::ScaleLossGradOpHandle(
        result->CreateEmptyNode("scale_loss_grad", ir::Node::Type::kOperation),
        loss_scale,
        local_scopes_[i],
        places_[i],
        dev_ctx,
        dtype);
    result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

    // FIXME: Currently ScaleLossGradOp only use device_count as scale
    // factor. So it does not depend on any other operators.
    // VarHandle *loss = GetVarHandle(loss_var_name, place);
    // loss->pending_ops_.emplace_back(op_handle);
    // op_handle->inputs_.emplace_back(loss);

    CreateOpOutput(result,
                   op_handle,
                   result->CreateVarNode(out_var_node->Var(),
                                         out_var_node->GetVarNodeBlockId()),
                   places_[i],
                   i);
  }
}

void MultiDevSSAGraphBuilderBase::CreateComputationalOps(
    ir::Graph *result, ir::Node *node, size_t num_places) const {
  for (size_t scope_idx = 0; scope_idx < num_places; ++scope_idx) {
    auto p = places_[scope_idx];
    auto s = local_scopes_[scope_idx];
    result->Get<GraphOps>(kGraphOps).emplace_back(
        new details::ComputationOpHandle(
            result->CreateOpNode(node->Op()), s, p, scope_idx));
    CreateOpHandleIOs(result, node, scope_idx);
  }
}

bool MultiDevSSAGraphBuilderBase::IsScaleLossOp(ir::Node *node) const {
  return !loss_var_name_.empty() && node->Op() &&
         PADDLE_GET_CONST(
             int,
             node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
             (static_cast<int>(OpRole::kBackward) |
              static_cast<int>(OpRole::kLoss));
}

bool MultiDevSSAGraphBuilderBase::IsSparseGradient(
    const std::string &og) const {
  PADDLE_ENFORCE_NE(all_vars_.count(og),
                    0,
                    platform::errors::InvalidArgument(
                        "Can not find Var(%s) in VarDescs "
                        "Paddle Can not add Collective OP for Var(%s).",
                        og,
                        og));
  return all_vars_.at(og)->GetType() == proto::VarType::SELECTED_ROWS;
}

void MultiDevSSAGraphBuilderBase::CreateIsolatedVarNode(
    ir::Graph *graph, ir::Node *var_node) const {
  for (size_t i = 0; i < places_.size(); ++i) {
    VLOG(10) << "Create isolated var node " << var_node->Name() << " at device "
             << i;
    CreateOrGetLatestVarHandle(graph, var_node, places_[i], i);
  }
}

void SetOpInputsAllPlaces(ir::Graph *result, ir::Node *node, int num_places) {
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
  for (ir::Node *input : node->inputs) {
    details::VarHandle *var = nullptr;
    for (int place_offset = 0; place_offset < num_places; ++place_offset) {
      auto &var_holders =
          result->Get<details::GraphVars>(details::kGraphVars)[place_offset];
      auto &var_holder = var_holders[input->Name()];
      if (!var_holder.empty()) {
        var = *var_holder.rbegin();
        op_handle->AddInput(var);
      }
    }
  }
}

std::unordered_set<std::string> &MultiDevSSAGraphBuilder() {
  static std::unordered_set<std::string> regs;
  return regs;
}

static int MultiDevSSAGraphBuilderRegister(const std::string &builder_mode) {
  MultiDevSSAGraphBuilder().insert(builder_mode);
  return 0;
}

}  // namespace paddle::framework::ir

#define REGISTER_MULTI_DEVICES_PASS(pass_name, pass_class)                \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      _reg_ssa_graph_builder_##pass_name,                                 \
      "REGISTER_MULTI_DEVICES_PASS must be called in global namespace."); \
  int _reg_ssa_graph_builder_entry_##pass_name =                          \
      paddle::framework::ir::MultiDevSSAGraphBuilderRegister(#pass_name); \
  REGISTER_PASS(pass_name, pass_class)                                    \
      .RequirePassAttr(paddle::framework::ir::kLossVarName)               \
      .RequirePassAttr(paddle::framework::details::kPlaces)               \
      .RequirePassAttr(paddle::framework::details::kLocalScopes)          \
      .RequirePassAttr(paddle::framework::ir::kStrategy)                  \
      .RequirePassAttr(paddle::framework::details::kNRanks)

REGISTER_MULTI_DEVICES_PASS(all_reduce_mode_multi_devices_pass,
                            paddle::framework::ir::MultiDevSSAGraphBuilderBase);
