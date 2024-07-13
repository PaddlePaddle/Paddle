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

  for (ir::Node *node : sorted_ops) {
    // This op runs on all devices
    if (IsScaleLossOp(node)) {
      // user can customize loss@grad if not use_default_grad_scale_
      // InsertScaleLossGradOp(&result, node);
    } else {
      CreateComputationalOps(&result, node, places_.size());
    }
  }
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

std::vector<ir::Node *> MultiDevSSAGraphBuilderBase::SortOperations(
    const ir::Graph &graph) const {
  return ir::TopologySortOperations(graph);
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

void MultiDevSSAGraphBuilderBase::CreateIsolatedVarNode(
    ir::Graph *graph, ir::Node *var_node) const {
  for (size_t i = 0; i < places_.size(); ++i) {
    VLOG(10) << "Create isolated var node " << var_node->Name() << " at device "
             << i;
    CreateOrGetLatestVarHandle(graph, var_node, places_[i], i);
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
