// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/transfer_layout_pass.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/framework/ir/cutlass_teller.h"

#include "paddle/common/layout.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle::framework::ir {
namespace {

void InsertLayoutTransOp(ir::Graph *graph,
                         ir::Node *prev_node,
                         ir::Node *next_node,
                         phi::DataLayout from_layout,
                         phi::DataLayout to_layout,
                         framework::BlockDesc *block_desc,
                         std::unordered_map<ir::Node *, ir::Node *> *cache) {
  auto do_insert = [&](const std::string &in_var_name,
                       const std::string &out_var_name) {
    auto update_op_desc = [&](framework::OpDesc &desc,
                              const std::string &x_name,
                              const std::string &out_name) {
      desc.SetType("transfer_layout");
      desc.SetInput("X", {x_name});
      desc.SetOutput("Out", {out_name});
      desc.SetAttr("src_layout", static_cast<int>(from_layout));
      desc.SetAttr("dst_layout", static_cast<int>(to_layout));
      desc.Flush();
    };
    PADDLE_ENFORCE_NOT_NULL(
        block_desc,
        common::errors::PreconditionNotMet(
            "BlockDesc should not be null when inserting layout transfer op."));
    if (cache->count(prev_node) == 0) {
      framework::OpDesc op_desc(block_desc);
      update_op_desc(op_desc, in_var_name, out_var_name);
      auto *op_node = graph->CreateOpNode(&op_desc);
      auto *op_out_var_desc = block_desc->Var(out_var_name);

      op_out_var_desc->SetPersistable(false);
      op_out_var_desc->SetDataType(prev_node->Var()->GetDataType());
      auto to_shape = prev_node->Var()->GetShape();
      if (from_layout == phi::DataLayout::kNCHW) {
        auto n = to_shape[0];
        auto c = to_shape[1];
        auto h = to_shape[2];
        auto w = to_shape[3];
        op_out_var_desc->SetShape({n, h, w, c});
      } else {
        auto n = to_shape[0];
        auto h = to_shape[1];
        auto w = to_shape[2];
        auto c = to_shape[3];
        op_out_var_desc->SetShape({n, c, h, w});
      }

      auto *op_out_var_node = graph->CreateVarNode(op_out_var_desc);
      IR_NODE_LINK_TO(op_node, op_out_var_node);
      cache->insert(std::make_pair(prev_node, op_out_var_node));
    }
    next_node->Op()->RenameInput(prev_node->Name(),
                                 cache->at(prev_node)->Name());
    IR_NODE_LINK_TO(prev_node, cache->at(prev_node)->inputs.front());
    IR_NODE_LINK_TO(cache->at(prev_node), next_node);

    IR_NODE_UNLINK(prev_node, next_node);
  };

  if (from_layout == phi::DataLayout::kNCHW &&
      to_layout == phi::DataLayout::kNHWC) {
    auto in_var_name = prev_node->Var()->Name();
    auto out_var_name = in_var_name + "_nchw_to_nhwc";
    do_insert(in_var_name, out_var_name);
  } else if (from_layout == phi::DataLayout::kNHWC &&
             to_layout == phi::DataLayout::kNCHW) {
    auto in_var_name = prev_node->Var()->Name();
    auto out_var_name = in_var_name + "_nhwc_to_nchw";
    do_insert(in_var_name, out_var_name);
  }
}

}  // namespace

void TransferLayoutPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      common::errors::PreconditionNotMet("graph should not be nullptr."));
  FusePassBase::Init("fused_conv2d_add_act_layout_transfer", graph);
  auto *scope = param_scope();

  // float16 for all(cutlass cudnn), float32 for cutlass.
  // why?
  // In the case of cudnn nhwc fp32, performance degradation will occur
  bool is_fp16_precision =
      static_cast<phi::DataType>(Get<int>("model_precision")) ==
          phi::DataType::FLOAT16 ||
      Get<bool>("enable_gpu_mixed");

  bool cutlass_enable = Get<bool>("use_cutlass");

  if (!is_fp16_precision && !cutlass_enable) return;

  PADDLE_ENFORCE_EQ(graph->IsMainGraph(),
                    true,
                    common::errors::InvalidArgument(
                        "the graph should be main graph when applying "
                        "transfer_layout_pass"));

  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::Fatal("scope must not be nullptr when applying "
                            "transfer_layout_pass"));

  // Not support multiple block now.
  std::unordered_map<ir::Node *, ir::Node *> cache;
  auto op_nodes = TopologySortOperations(*graph);
  auto iter = op_nodes.cbegin();
  auto *block_desc = (*iter)->Op()->Block();

  // Process multiple fused_conv2d_add_act shares weight.
  std::unordered_map<std::string, std::vector<ir::Node *>> weights_shared;

  // Used to control the insertion of transfer_layout op.
  std::unordered_set<ir::Node *> vars_shape_nhwc;

  // Only support fused_conv2d_add_act now.
  std::string target_op_type = "fused_conv2d_add_act";
  std::unordered_set<ir::Node *> valid_ops;

  // Determine if this fused_conv2d_add_act can run in cuDNN's NHWC mode,
  // will not set or change any attribute in op_desc
  auto cuDNNIsValid = [&](ir::Node *op_node) -> bool {
    auto filter_names = op_node->Op()->Input("Filter");
    constexpr int CUDNN_ALIGNMENT = 8;
    // If filter's channel is not multiple of CUDNN_ALIGNMENT,
    // fused_conv2d_add_act not run at nhwc.
    for (const auto &filter_name : filter_names) {
      auto *filter_var = scope->FindLocalVar(filter_name);
      const auto &filter_tensor = filter_var->Get<phi::DenseTensor>();
      PADDLE_ENFORCE_EQ(
          filter_tensor.dims().size(),
          4UL,
          common::errors::InvalidArgument(
              "The 'Filter' tensor should have 4 dimensions, but received %d.",
              filter_tensor.dims().size()));
      int oc = static_cast<int>(filter_tensor.dims()[0]);
      int ic = static_cast<int>(filter_tensor.dims()[1]);
      bool cudnn_can_support =
          oc % CUDNN_ALIGNMENT == 0 && ic % CUDNN_ALIGNMENT == 0;
      if (!cudnn_can_support) {
        return false;
      }
    }
    return true;
  };

  auto CutlassIsValid = [&](ir::Node *op_node) -> bool {
    auto op_desc = op_node->Op();
    bool use_cudnn = true;
    if (op_desc->HasAttr("use_cudnn")) {
      use_cudnn = op_desc->GetAttrIfExists<bool>("use_cudnn");
    }
    bool cutlass_enable = Get<bool>("use_cutlass");
    return !use_cudnn && cutlass_enable;
  };

  for (auto *op_node : op_nodes) {
    PADDLE_ENFORCE_EQ(op_node->IsOp(),
                      true,
                      common::errors::InvalidArgument(
                          "The node should be an operation, but it's not."));
    // some common check.
    if (op_node->Op()->Type() != target_op_type) {
      continue;
    }

    auto data_format =
        op_node->Op()->GetAttrIfExists<std::string>("data_format");
    if (data_format != "NCHW") {
      continue;
    }

    auto filter_names = op_node->Op()->Input("Filter");
    for (const auto &filter_name : filter_names) {
      weights_shared[filter_name].push_back(op_node);
    }

    if (cuDNNIsValid(op_node) || CutlassIsValid(op_node)) {
      valid_ops.insert(op_node);
    }
  }

  // The target operators that share weights either run nhwc or not at all.
  for (auto *op_node : op_nodes) {
    PADDLE_ENFORCE_EQ(op_node->IsOp(),
                      true,
                      common::errors::InvalidArgument(
                          "The node should be an operation, but it's not."));
    if (valid_ops.count(op_node)) {
      auto filter_names = op_node->Op()->Input("Filter");
      for (const auto &filter_name : filter_names) {
        if (std::any_of(weights_shared[filter_name].begin(),
                        weights_shared[filter_name].end(),
                        [&valid_ops](ir::Node *node) {
                          return valid_ops.count(node) == 0;
                        })) {
          for (auto *node : weights_shared[filter_name]) {
            valid_ops.erase(node);
          }
        }
      }
    }
  }

  // Insert transfer_layout op
  for (auto *op_node : op_nodes) {
    PADDLE_ENFORCE_EQ(op_node->IsOp(),
                      true,
                      common::errors::InvalidArgument(
                          "The node should be an operation, but it's not."));

    if (valid_ops.count(op_node)) {
      auto *op_desc = op_node->Op();

      if (CutlassIsValid(op_node)) {
        // fused_conv2d_add_act must have this attribute because of signature.
        if (!op_desc->HasAttr("fuse_alpha")) {
          op_desc->SetAttr("fuse_alpha", 0.f);
        }
      } else if (cuDNNIsValid(op_node)) {
        if (op_desc->HasAttr("use_cudnn")) {
          op_desc->SetAttr("use_cudnn", true);
        }
      }
      op_desc->SetAttr("data_format", std::string{"NHWC"});
      op_desc->Flush();

      // transfer weights
      auto filter_names = op_desc->Input("Filter");
      for (const auto &filter_name : filter_names) {
        auto *filter_var = scope->FindLocalVar(filter_name);
        auto *filter_tensor = filter_var->GetMutable<phi::DenseTensor>();
        if (filter_tensor->layout() == phi::DataLayout::kNHWC) {
          continue;
        }
        phi::DenseTensor temp_tensor;

        framework::TransDataLayout(phi::DataLayout::kNCHW,
                                   phi::DataLayout::kNHWC,
                                   phi::CPUPlace{},
                                   *filter_tensor,
                                   &temp_tensor);
        *filter_tensor = temp_tensor;

        auto op_inputs = op_node->inputs;
        for (auto *in_var_node : op_inputs) {
          PADDLE_ENFORCE_EQ(
              in_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "The node should be a variable, but it's not."));
          if (in_var_node->Var()->Persistable() &&
              in_var_node->Var()->Name() == filter_name) {
            auto from_shape = in_var_node->Var()->GetShape();
            in_var_node->Var()->SetShape(
                {from_shape[0], from_shape[2], from_shape[3], from_shape[1]});
          }
        }
      }

      // transfer outputs
      auto op_outputs = op_node->outputs;
      for (auto *out_var_node : op_outputs) {
        PADDLE_ENFORCE_EQ(out_var_node->IsVar(),
                          true,
                          common::errors::InvalidArgument(
                              "The node should be a variable, but it's not."));

        if (out_var_node->Var()->Persistable()) continue;

        auto from_shape = out_var_node->Var()->GetShape();
        out_var_node->Var()->SetShape(
            {from_shape[0], from_shape[2], from_shape[3], from_shape[1]});
        vars_shape_nhwc.insert(out_var_node);
      }

      // Insert transfer_layout for intermediate var.
      auto op_inputs = op_node->inputs;
      for (auto *in_var_node : op_inputs) {
        PADDLE_ENFORCE_EQ(in_var_node->IsVar(),
                          true,
                          common::errors::InvalidArgument(
                              "The node should be a variable, but it's not."));

        if (in_var_node->Var()->Persistable()) continue;
        if (vars_shape_nhwc.count(in_var_node)) continue;

        InsertLayoutTransOp(graph,
                            in_var_node,
                            op_node,
                            phi::DataLayout::kNCHW,
                            phi::DataLayout::kNHWC,
                            block_desc,
                            &cache);
      }
    } else {
      auto op_inputs = op_node->inputs;
      for (auto *in_var_node : op_inputs) {
        PADDLE_ENFORCE_EQ(in_var_node->IsVar(),
                          true,
                          common::errors::InvalidArgument(
                              "The node should be a variable, but it's not."));

        if (vars_shape_nhwc.count(in_var_node)) {
          InsertLayoutTransOp(graph,
                              in_var_node,
                              op_node,
                              phi::DataLayout::kNHWC,
                              phi::DataLayout::kNCHW,
                              block_desc,
                              &cache);
        }
      }
    }
  }
  AddStatis(static_cast<int>(valid_ops.size()));
}

}  // namespace paddle::framework::ir

REGISTER_PASS(transfer_layout_pass, paddle::framework::ir::TransferLayoutPass);
