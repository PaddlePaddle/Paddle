// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/trt_support_nhwc_pass.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/phi/common/data_type.h"

namespace paddle::framework::ir {

namespace {

void DoInsertTransposeOp(ir::Graph *graph,
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
                              const std::string &out_name,
                              const std::vector<int> &axis_attr) {
      desc.SetType("transpose");
      desc.SetInput("X", {x_name});
      desc.SetOutput("Out", {out_name});
      desc.SetAttr("axis", axis_attr);
      desc.SetAttr("use_mkldnn", false);
      desc.SetAttr("data_format", std::string{"AnyLayout"});
      desc.SetAttr("use_quantizer", false);
      desc.SetAttr("mkldnn_data_type", std::string{"float32"});
      desc.Flush();
    };
    PADDLE_ENFORCE_NOT_NULL(block_desc,
                            common::errors::InvalidArgument(
                                "During the trt_support_nhwc_pass, the "
                                "block description should not be null."));
    if (cache->count(prev_node) == 0) {
      framework::OpDesc op_desc(block_desc);
      if (from_layout == phi::DataLayout::kNCHW) {
        update_op_desc(op_desc, in_var_name, out_var_name, {0, 2, 3, 1});
      } else if (from_layout == phi::DataLayout::kNHWC) {
        update_op_desc(op_desc, in_var_name, out_var_name, {0, 3, 1, 2});
      }
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
      } else if (from_layout == phi::DataLayout::kNHWC) {
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

bool ModelLayoutIsNHWC(const std::vector<ir::Node *> &op_nodes) {
  for (auto *op_node : op_nodes) {
    if (op_node->IsOp()) {
      auto *op_desc = op_node->Op();
      std::string data_format;
      if (op_desc->HasAttr("data_format")) {
        data_format = op_desc->GetAttrIfExists<std::string>("data_format");
      } else if (op_desc->HasAttr("data_layout")) {
        data_format = op_desc->GetAttrIfExists<std::string>("data_layout");
      }
      if (data_format == "NHWC") {
        return true;
      }
    }
  }
  return false;
}

// Do additional check if OP's weight is not persistable
typedef std::string OP_NAME;
typedef std::string WEIGHT_NAME;
typedef std::unordered_map<OP_NAME, WEIGHT_NAME> OP_WEIGHT_NAME;
bool IsWeight(ir::Node *op_node,
              ir::Node *var_node,
              const OP_WEIGHT_NAME &op_weight_pair) {
  if (var_node->Var()->Persistable()) return true;
  auto *op_desc = op_node->Op();
  std::string op_type = op_desc->Type();
  std::string var_name = var_node->Var()->Name();
  if (op_weight_pair.count(op_type)) {
    if (var_name ==
        op_desc->Input(op_weight_pair.find(op_type)->second).front()) {
      return true;
    }
  }
  return false;
}

}  // namespace

void TrtSupportNHWCPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::PreconditionNotMet(
                              "During the trt_support_nhwc_pass, the graph "
                              "should not be null."));
  FusePassBase::Init("trt_support_nhwc_pass", graph);
  auto *scope = param_scope();

  auto op_nodes = TopologySortOperations(*graph);

  if (!ModelLayoutIsNHWC(op_nodes)) {
    return;
  }

  //
  //
  // TODO(liuyuanle): Add other op if needed!
  //
  //
  std::unordered_set<std::string> need_trans_weights{"prelu"};
  std::unordered_set<std::string> not_trans_weights{"conv2d",
                                                    "pool2d",
                                                    "batch_norm",
                                                    "bilinear_interp",
                                                    "bilinear_interp_v2",
                                                    "nearest_interp",
                                                    "nearest_interp_v2"};
  // Op's weight could be temporary variable, so we save the name of OP's weight
  // input
  OP_WEIGHT_NAME op_weight_pair{{"conv2d", "Filter"}};
  // Ops must run under the original layout even though it has
  // data_format/data_layout attribute, otherwise it will be very troublesome!
  std::unordered_set<std::string> must_original_layout_ops{
      "affine_channel", "softmax", "temporal_shift"};
  // OPs unrelated to layout are consistent according to the layout of input
  // var！
  std::unordered_set<std::string> any_layout_ops{"relu", "elementwise_add"};
  //
  //
  // TODO(liuyuanle): Add other op if needed!
  //
  //

  // Ops with "data_format" or "data_layout" attribute value of "NHWC"
  std::unordered_set<ir::Node *> transposed_ops;
  std::unordered_set<ir::Node *> vars_to_nchw;

  std::unordered_map<ir::Node *, ir::Node *> cache;

  // Not support multiple block now
  auto iter = op_nodes.cbegin();
  auto *block_desc = (*iter)->Op()->Block();

  for (auto *op_node : op_nodes) {
    PADDLE_ENFORCE_EQ(op_node->IsOp(),
                      true,
                      common::errors::InvalidArgument(
                          "op_node->IsOp() is False, which means that "
                          "%p may be an invalid option.",
                          op_node));
    auto *op_desc = op_node->Op();

    std::string data_format;
    if (op_desc->HasAttr("data_format")) {
      data_format = op_desc->GetAttrIfExists<std::string>("data_format");
    } else if (op_desc->HasAttr("data_layout")) {
      data_format = op_desc->GetAttrIfExists<std::string>("data_layout");
    }

    bool input_shape_4{true};
    auto op_inputs = op_node->inputs;
    for (auto *in_var_node : op_inputs) {
      PADDLE_ENFORCE_EQ(in_var_node->IsVar(),
                        true,
                        common::errors::InvalidArgument(
                            "in_var_node->IsVar() is False, which means that "
                            "inputs may be not a valid variable."));
      if (IsWeight(op_node, in_var_node, op_weight_pair)) continue;

      auto input_shape = in_var_node->Var()->GetShape();
      input_shape_4 &= (input_shape.size() == 4);
    }

    if (data_format != "NHWC" || !input_shape_4 ||
        any_layout_ops.count(op_desc->Type()) ||
        must_original_layout_ops.count(op_desc->Type())) {
      continue;
    }
    // Transpose NHWC --> NCHW
    //
    // Update current op
    transposed_ops.insert(op_node);
    if (op_desc->HasAttr("data_format")) {
      op_desc->SetAttr("data_format", std::string{"NCHW"});
      op_desc->Flush();
    } else if (op_desc->HasAttr("data_layout")) {
      op_desc->SetAttr("data_layout", std::string{"NCHW"});
      op_desc->Flush();
    }

    auto UpdateOutputVars = [&] {
      // Update output var of current op
      auto op_outputs = op_node->outputs;
      for (auto *out_var_node : op_outputs) {
        PADDLE_ENFORCE_EQ(
            out_var_node->IsVar(),
            true,
            common::errors::InvalidArgument(
                "out_var_node->IsVar() is False, which means that "
                "outputs may be not a valid variable."));
        if (out_var_node->Var()->Persistable()) continue;

        auto from_shape = out_var_node->Var()->GetShape();
        if (from_shape.size() == 4) {
          out_var_node->Var()->SetShape(
              {from_shape[0], from_shape[3], from_shape[1], from_shape[2]});
          vars_to_nchw.insert(out_var_node);
        }
      }
    };

    if (not_trans_weights.count(op_desc->Type())) {
      UpdateOutputVars();
    } else if (need_trans_weights.count(op_desc->Type())) {
      std::vector<std::string> weights;
      if (op_desc->Type() == "prelu") {
        weights.push_back("Alpha");
      }
      auto UpdateWeightVars = [&] {
        for (auto const &weight : weights) {
          // transfer weights
          auto weight_names = op_desc->Input(weight);
          for (const auto &weight_name : weight_names) {
            auto *weight_var = scope->FindLocalVar(weight_name);
            auto *weight_tensor = weight_var->GetMutable<phi::DenseTensor>();
            if (weight_tensor->dims().size() == 4) {
              phi::DenseTensor temp_tensor = *weight_tensor;
              weight_tensor->clear();

              framework::TransDataLayout(phi::DataLayout::kNHWC,
                                         phi::DataLayout::kNCHW,
                                         phi::CPUPlace{},
                                         temp_tensor,
                                         weight_tensor);
            }
          }
          auto op_inputs = op_node->inputs;
          for (auto *in_var_node : op_inputs) {
            PADDLE_ENFORCE_EQ(
                in_var_node->IsVar(),
                true,
                common::errors::InvalidArgument(
                    "in_var_node->IsVar() is False, which means that "
                    "inputs may be not a valid variable."));
            if (in_var_node->Var()->Persistable()) {
              if (std::find(weight_names.cbegin(),
                            weight_names.cend(),
                            in_var_node->Var()->Name()) !=
                  weight_names.cend()) {
                auto from_shape = in_var_node->Var()->GetShape();
                in_var_node->Var()->SetShape({from_shape[0],
                                              from_shape[2],
                                              from_shape[3],
                                              from_shape[1]});
              }
            }
          }
        }
      };
      UpdateWeightVars();
      UpdateOutputVars();
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "During the trt_support_nhwc_pass, %s op not supported. Please "
          "update the supported op lists.",
          op_desc->Type()));
    }
  }

  auto ProcessAnyLayoutOps = [&] {
    // Process any layout ops
    for (auto *op_node : op_nodes) {
      PADDLE_ENFORCE_EQ(op_node->IsOp(),
                        true,
                        common::errors::InvalidArgument(
                            "op_node->IsOp() is False, which means that "
                            "%p may be an invalid option.",
                            op_node));
      auto op_inputs = op_node->inputs;
      for (auto *in_var_node : op_inputs) {
        PADDLE_ENFORCE_EQ(in_var_node->IsVar(),
                          true,
                          common::errors::InvalidArgument(
                              "in_var_node->IsVar() is False, which means that "
                              "inputs may be not a valid variable."));
        if (transposed_ops.count(op_node)) continue;

        if (vars_to_nchw.count(in_var_node) &&
            any_layout_ops.count(op_node->Op()->Type())) {
          transposed_ops.insert(op_node);
          // Update output var of current op
          auto op_outputs = op_node->outputs;
          for (auto *out_var_node : op_outputs) {
            PADDLE_ENFORCE_EQ(
                out_var_node->IsVar(),
                true,
                common::errors::InvalidArgument(
                    "out_var_node->IsVar() is False, which means that "
                    "outputs may be not a valid variable."));
            if (out_var_node->Var()->Persistable()) continue;

            auto from_shape = out_var_node->Var()->GetShape();
            if (from_shape.size() == 4) {
              out_var_node->Var()->SetShape(
                  {from_shape[0], from_shape[3], from_shape[1], from_shape[2]});
              vars_to_nchw.insert(out_var_node);
            }
          }
        }
      }
    }
  };
  ProcessAnyLayoutOps();

  auto InsertTransposeOp = [&] {
    // Insert transpose op
    for (auto *op_node : op_nodes) {
      PADDLE_ENFORCE_EQ(op_node->IsOp(),
                        true,
                        common::errors::InvalidArgument(
                            "op_node->IsOp() is False, which means that "
                            "%p may be an invalid option.",
                            op_node));

      if (transposed_ops.count(op_node)) {
        auto op_inputs = op_node->inputs;
        for (auto *in_var_node : op_inputs) {
          PADDLE_ENFORCE_EQ(
              in_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "in_var_node->IsVar() is False, which means that "
                  "inputs may be not a valid variable."));

          if (IsWeight(op_node, in_var_node, op_weight_pair)) continue;
          if (vars_to_nchw.count(in_var_node)) continue;

          DoInsertTransposeOp(graph,
                              in_var_node,
                              op_node,
                              phi::DataLayout::kNHWC,
                              phi::DataLayout::kNCHW,
                              block_desc,
                              &cache);
        }
      } else {
        auto op_inputs = op_node->inputs;
        for (auto *in_var_node : op_inputs) {
          PADDLE_ENFORCE_EQ(
              in_var_node->IsVar(),
              true,
              common::errors::InvalidArgument(
                  "in_var_node->IsVar() is False, which means that "
                  "inputs may be not a valid variable."));

          if (vars_to_nchw.count(in_var_node)) {
            DoInsertTransposeOp(graph,
                                in_var_node,
                                op_node,
                                phi::DataLayout::kNCHW,
                                phi::DataLayout::kNHWC,
                                block_desc,
                                &cache);
          }
        }
      }
    }
  };
  InsertTransposeOp();

  AddStatis(transposed_ops.size());
}

}  // namespace paddle::framework::ir

REGISTER_PASS(trt_support_nhwc_pass, paddle::framework::ir::TrtSupportNHWCPass);
