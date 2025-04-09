/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir {

struct Layers {
 public:
  const ProgramDesc& main_program() { return program_; }

  BlockDesc* Block() { return program_.MutableBlock(0); }

  VarDesc* data(std::string name,
                std::vector<int64_t> shape = {},
                bool is_persistable = false,
                proto::VarType::Type data_type = proto::VarType::FP32) {
    return dense_tensor(name, shape, is_persistable, data_type);
  }

  VarDesc* conv2d(VarDesc* input,
                  VarDesc* filter,
                  VarDesc* bias,
                  int groups = 1,
                  std::vector<int> strides = {1, 1},
                  std::vector<int> paddings = {0, 0},
                  std::string padding_algorithm = "EXPLICIT",
                  std::vector<int> dilations = {1, 1},
                  std::string data_format = "NCHW",
                  bool use_cudnn = false) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("conv2d");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Filter", {filter->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Output", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    op->SetAttr("groups", groups);
    op->SetAttr("strides", strides);
    op->SetAttr("paddings", paddings);
    op->SetAttr("padding_algorithm", padding_algorithm);
    op->SetAttr("dilations", dilations);
    op->SetAttr("data_format", data_format);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* conv2d_transpose(VarDesc* input,
                            VarDesc* filter,
                            VarDesc* bias,
                            int groups = 1,
                            std::vector<int> strides = {1, 1},
                            std::vector<int> paddings = {0, 0},
                            std::string padding_algorithm = "EXPLICIT",
                            std::vector<int> dilations = {1, 1},
                            std::string data_format = "NCHW") {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("conv2d_transpose");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Filter", {filter->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Output", {out->Name()});
    op->SetAttr("groups", groups);
    op->SetAttr("strides", strides);
    op->SetAttr("paddings", paddings);
    op->SetAttr("padding_algorithm", padding_algorithm);
    op->SetAttr("dilations", dilations);
    op->SetAttr("data_format", data_format);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* depthwise_conv2d(VarDesc* input,
                            VarDesc* filter,
                            VarDesc* bias,
                            bool use_cudnn) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("depthwise_conv2d");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Filter", {filter->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* pool2d(VarDesc* x,
                  bool use_cudnn,
                  const AttributeMap* attrs = nullptr) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("pool2d");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    if (attrs) {
      for (auto& iter : *attrs) {
        op->SetAttr(iter.first, iter.second);
      }
    }
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* squeeze2(VarDesc* x,
                    const std::vector<int> axes = {-1},
                    bool with_xshape = false) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("squeeze2");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("axes", axes);
    if (with_xshape) {
      VarDesc* xshape = dense_tensor(unique_name());
      op->SetOutput("XShape", {xshape->Name()});
    }
    return out;
  }

  VarDesc* unsqueeze2(VarDesc* x, const std::vector<int> axes = {-1}) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("unsqueeze2");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("axes", axes);
    return out;
  }

  VarDesc* relu(VarDesc* x, VarDesc* out = nullptr) {
    return unary_op("relu", x, out);
  }

  VarDesc* gelu(VarDesc* x, VarDesc* out = nullptr, bool approximate = true) {
    AttributeMap attrs;
    attrs["approximate"] = approximate;
    return unary_op("gelu", x, out, &attrs);
  }

  VarDesc* sigmoid(VarDesc* x, VarDesc* out = nullptr) {
    return unary_op("sigmoid", x, out);
  }

  VarDesc* tanh(VarDesc* x, VarDesc* out = nullptr) {
    return unary_op("tanh", x, out);
  }

  VarDesc* c_identity(VarDesc* x, VarDesc* out = nullptr, int ring_id = -1) {
    AttributeMap attrs;
    attrs["ring_id"] = ring_id;
    return unary_op("c_identity", x, out, &attrs);
  }

  VarDesc* c_allreduce_sum(VarDesc* x,
                           VarDesc* out = nullptr,
                           int ring_id = -1) {
    AttributeMap attrs;
    attrs["ring_id"] = ring_id;
    return unary_op("c_allreduce_sum", x, out, &attrs);
  }

  VarDesc* fc(VarDesc* input,
              VarDesc* w,
              VarDesc* bias,
              int in_num_col_dims = 1,
              std::string activation_type = "") {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("fc");
    op->SetInput("Input", {input->Name()});
    op->SetInput("W", {w->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("in_num_col_dims", in_num_col_dims);
    op->SetAttr("activation_type", activation_type);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  void lstm(VarDesc* input,
            VarDesc* w,
            VarDesc* bias,
            VarDesc* cell,
            VarDesc* batch_gate,
            VarDesc* hidden,
            VarDesc* batch_cell_pre_act,
            VarDesc* h0 = nullptr,
            VarDesc* c0 = nullptr,
            bool use_peepholes = true,
            bool is_reverse = false,
            std::string gate_activation = "sigmoid",
            std::string cell_activation = "tanh",
            std::string candidate_activation = "tanh") {
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("lstm");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Weight", {w->Name()});
    op->SetInput("Bias", {bias->Name()});
    if (h0) {
      op->SetInput("H0", {h0->Name()});
    }
    if (c0) {
      op->SetInput("C0", {c0->Name()});
    }
    op->SetOutput("Hidden", {hidden->Name()});
    op->SetOutput("Cell", {cell->Name()});
    op->SetOutput("BatchGate", {batch_gate->Name()});
    op->SetOutput("BatchCellPreAct", {batch_cell_pre_act->Name()});
    op->SetAttr("use_peepholes", use_peepholes);
    op->SetAttr("is_reverse", is_reverse);
    op->SetAttr("gate_activation", gate_activation);
    op->SetAttr("cell_activation", cell_activation);
    op->SetAttr("candidate_activation", candidate_activation);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
  }

  void gru(VarDesc* input,
           VarDesc* w,
           VarDesc* bias,
           VarDesc* batch_gate,
           VarDesc* batch_reset_hidden_prev,
           VarDesc* batch_hidden,
           VarDesc* hidden,
           VarDesc* h0 = nullptr,
           bool origin_mode = false,
           bool is_reverse = false,
           std::string activation = "tanh",
           std::string gate_activation = "sigmoid") {
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("gru");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Weight", {w->Name()});
    op->SetInput("Bias", {bias->Name()});
    if (h0) {
      op->SetInput("H0", {h0->Name()});
    }
    op->SetOutput("BatchGate", {batch_gate->Name()});
    op->SetOutput("BatchResetHiddenPrev", {batch_reset_hidden_prev->Name()});
    op->SetOutput("BatchHidden", {batch_hidden->Name()});
    op->SetOutput("Hidden", {hidden->Name()});
    op->SetAttr("origin_mode", origin_mode);
    op->SetAttr("is_reverse", is_reverse);
    op->SetAttr("activation", activation);
    op->SetAttr("gate_activation", gate_activation);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
  }

  VarDesc* mul(VarDesc* x,
               VarDesc* y,
               VarDesc* out = nullptr,
               int x_num_col_dims = 1,
               int y_num_col_dims = 1,
               bool use_mkldnn = false) {
    AttributeMap attrs;
    attrs["x_num_col_dims"] = x_num_col_dims;
    attrs["y_num_col_dims"] = y_num_col_dims;
    attrs["use_mkldnn"] = use_mkldnn;
    return binary_op("mul", x, y, out, &attrs);
  }

  VarDesc* elementwise_add(VarDesc* x,
                           VarDesc* y,
                           VarDesc* out = nullptr,
                           int axis = -1,
                           bool use_mkldnn = false) {
    AttributeMap attrs;
    attrs["axis"] = axis;
    attrs["use_mkldnn"] = use_mkldnn;
    return binary_op("elementwise_add", x, y, out, &attrs);
  }

  VarDesc* elementwise_mul(VarDesc* x,
                           VarDesc* y,
                           VarDesc* out = nullptr,
                           const AttributeMap* attrs = nullptr) {
    return binary_op("elementwise_mul", x, y, out, attrs);
  }

  VarDesc* elementwise_div(VarDesc* x,
                           VarDesc* y,
                           VarDesc* out = nullptr,
                           const AttributeMap* attrs = nullptr) {
    return binary_op("elementwise_div", x, y, out, attrs);
  }

  VarDesc* dropout(VarDesc* x,
                   float dropout_prob,
                   std::string dropout_implementation) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("dropout");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("is_test", true);
    op->SetAttr("dropout_prob", dropout_prob);
    op->SetAttr("dropout_implementation", dropout_implementation);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* concat(std::vector<VarDesc*> inputs, int axis = -1) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("concat");
    std::vector<std::string> input_names(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_names[i] = inputs[i]->Name();
    }
    op->SetInput("X", input_names);
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("axis", axis);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  std::vector<VarDesc*> layer_norm(VarDesc* x,
                                   VarDesc* scale = nullptr,
                                   VarDesc* bias = nullptr) {
    VarDesc* y = dense_tensor(unique_name());
    VarDesc* mean = dense_tensor(unique_name());
    VarDesc* variance = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("layer_norm");
    op->SetInput("X", {x->Name()});
    if (scale) {
      op->SetInput("Scale", {scale->Name()});
    }
    if (bias) {
      op->SetInput("Bias", {bias->Name()});
    }
    op->SetOutput("Y", {y->Name()});
    op->SetOutput("Mean", {mean->Name()});
    op->SetOutput("Variance", {variance->Name()});
    op->SetAttr("epsilon", static_cast<float>(1E-05));
    op->SetAttr("begin_norm_axis", static_cast<int>(1));
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    std::vector<VarDesc*> outs = {y, mean, variance};
    return outs;
  }

  std::vector<VarDesc*> split(VarDesc* x,
                              int num_or_section = 0,
                              int axis = 0,
                              std::vector<int> sections = {-1}) {
    int out_num = num_or_section;
    if (num_or_section == 0) {
      out_num = sections.size();
    }
    std::vector<VarDesc*> outs(out_num);
    for (int i = 0; i < out_num; i++) {
      outs[i] = dense_tensor(unique_name());
    }
    std::vector<std::string> out_names(out_num);
    for (int i = 0; i < out_num; i++) {
      out_names[i] = outs[i]->Name();
    }
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("split");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", out_names);
    if (num_or_section == 0) {
      op->SetAttr("sections", sections);
    } else {
      op->SetAttr("num_or_section", num_or_section);
    }
    op->SetAttr("axis", axis);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return outs;
  }

  VarDesc* assign(VarDesc* x) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("assign");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* matmul(VarDesc* x,
                  VarDesc* y,
                  VarDesc* alpha = nullptr,
                  bool transpose_x = false,
                  bool transpose_y = false) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("matmul");
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("transpose_X", transpose_x);
    op->SetAttr("transpose_Y", transpose_y);
    op->SetAttr("alpha", 1.0f);
    return out;
  }

  VarDesc* clip(VarDesc* x, VarDesc* min, VarDesc* max) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("clip");
    op->SetInput("X", {x->Name()});
    op->SetInput("Min", {min->Name()});
    op->SetInput("Max", {max->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* matmul_v2(VarDesc* x,
                     VarDesc* y,
                     VarDesc* alpha = nullptr,
                     bool trans_x = false,
                     bool trans_y = false) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("matmul_v2");
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("trans_x", trans_x);
    op->SetAttr("trans_y", trans_y);
    return out;
  }

  VarDesc* transpose2(VarDesc* x,
                      std::vector<int> axis,
                      bool with_xshape = false) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("transpose2");
    op->SetInput("X", {x->Name()});
    op->SetAttr("axis", axis);
    op->SetOutput("Out", {out->Name()});
    if (with_xshape) {
      VarDesc* xshape = dense_tensor(unique_name());
      op->SetOutput("XShape", {xshape->Name()});
    }
    return out;
  }

  VarDesc* reshape2(VarDesc* x,
                    std::vector<int> shape,
                    bool with_xshape = false) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("reshape2");
    op->SetInput("X", {x->Name()});
    op->SetAttr("shape", shape);
    op->SetOutput("Out", {out->Name()});
    if (with_xshape) {
      VarDesc* xshape = dense_tensor(unique_name());
      op->SetOutput("XShape", {xshape->Name()});
    }
    return out;
  }

  VarDesc* softmax(VarDesc* x, int axis) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("softmax");
    op->SetInput("X", {x->Name()});
    op->SetAttr("axis", axis);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* scale(VarDesc* x,
                 float scale = 1.,
                 float bias = 0.,
                 bool bias_after = true) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("scale");
    op->SetInput("X", {x->Name()});
    op->SetAttr("scale", scale);
    op->SetAttr("bias", bias);
    op->SetAttr("bias_after_scale", bias_after);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  std::vector<VarDesc*> batch_norm(VarDesc* x,
                                   VarDesc* scale,
                                   VarDesc* bias,
                                   VarDesc* mean,
                                   VarDesc* variance) {
    VarDesc* y = dense_tensor(unique_name());
    VarDesc* mean_out = dense_tensor(unique_name());
    VarDesc* variance_out = dense_tensor(unique_name());
    VarDesc* saved_mean = dense_tensor(unique_name());
    VarDesc* saved_variance = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("batch_norm");
    op->SetInput("X", {x->Name()});
    op->SetInput("Scale", {scale->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetInput("Mean", {mean->Name()});
    op->SetInput("Variance", {variance->Name()});
    op->SetOutput("Y", {y->Name()});
    op->SetOutput("MeanOut", {mean_out->Name()});
    op->SetOutput("VarianceOut", {variance_out->Name()});
    op->SetOutput("SavedMean", {saved_mean->Name()});
    op->SetOutput("SavedVariance", {saved_variance->Name()});
    op->SetAttr("epsilon", static_cast<float>(1e-5));
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    std::vector<VarDesc*> outs = {
        y, mean_out, variance_out, saved_mean, saved_variance};
    return outs;
  }

  VarDesc* embedding(VarDesc* x, VarDesc* weights) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("lookup_table");
    op->SetInput("Ids", {x->Name()});
    op->SetInput("W", {weights->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* while_loop(std::vector<VarDesc*> xs, VarDesc* cond = nullptr) {
    VarDesc* out = dense_tensor(unique_name());
    VarDesc* step_scopes = dense_tensor(unique_name());
    if (cond == nullptr) cond = dense_tensor(unique_name());

    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("while");
    std::vector<std::string> xs_names;
    for (auto& x : xs) xs_names.emplace_back(x->Name());
    op->SetInput("X", xs_names);
    op->SetInput("Condition", {cond->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetOutput("StepScopes", {step_scopes->Name()});
    op->SetAttr("sub_block", {program_.MutableBlock(0)});
    op->SetAttr("is_test", true);
    return out;
  }

  VarDesc* shape(VarDesc* input) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("shape");
    op->SetInput("Input", {input->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* slice(VarDesc* input,
                 std::vector<int> axes,
                 std::vector<int> starts,
                 std::vector<int> ends) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("slice");
    op->SetInput("Input", {input->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("axes", axes);
    op->SetAttr("starts", starts);
    op->SetAttr("ends", ends);
    return out;
  }

  VarDesc* fill_constant_batch_size_like(VarDesc* x,
                                         int dtype,
                                         int input_dim_idx,
                                         int output_dim_idx,
                                         std::vector<int> shape,
                                         float value) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("fill_constant_batch_size_like");
    op->SetInput("Input", {x->Name()});
    op->SetAttr("dtype", dtype);
    op->SetAttr("input_dim_idx", input_dim_idx);
    op->SetAttr("output_dim_idx", output_dim_idx);
    op->SetAttr("shape", shape);
    op->SetAttr("value", value);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  std::vector<VarDesc*> fused_multi_transformer(
      VarDesc* x,
      VarDesc* cache_kv,
      VarDesc* src_mask,
      VarDesc* qkv_w,
      VarDesc* qkv_bias,
      VarDesc* out_linear_w,
      VarDesc* out_linear_bias,
      VarDesc* ffn1_w,
      VarDesc* ffn1_bias,
      VarDesc* ffn2_w,
      VarDesc* ffn2_bias,
      VarDesc* ln_scale,
      VarDesc* ln_bias,
      VarDesc* ffn_ln_scale,
      VarDesc* ffn_ln_bias,
      float epsilon,
      float dropout_rate,
      VarDesc* time_stamp = nullptr,
      VarDesc* qkv_out_scale = nullptr,
      VarDesc* out_linear_out_scale = nullptr,
      VarDesc* ffn1_out_scale = nullptr,
      VarDesc* ffn2_out_scale = nullptr,
      std::vector<float> qkv_in_scale = {},
      std::vector<float> out_linear_in_scale = {},
      std::vector<float> ffn1_in_scale = {},
      std::vector<float> ffn2_in_scale = {}) {
    VarDesc* out = dense_tensor(unique_name());
    VarDesc* cache_kv_out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    std::string op_type = qkv_out_scale ? "fused_multi_transformer_int8"
                                        : "fused_multi_transformer";
    op->SetType(op_type);
    op->SetInput("X", {x->Name()});
    op->SetInput("CacheKV", {cache_kv->Name()});
    op->SetInput("SrcMask", {src_mask->Name()});
    op->SetInput("QKVW", {qkv_w->Name()});
    op->SetInput("QKVBias", {qkv_bias->Name()});
    op->SetInput("OutLinearW", {out_linear_w->Name()});
    op->SetInput("OutLinearBias", {out_linear_bias->Name()});
    op->SetInput("FFN1Weight", {ffn1_w->Name()});
    op->SetInput("FFN1Bias", {ffn1_bias->Name()});
    op->SetInput("FFN2Weight", {ffn2_w->Name()});
    op->SetInput("FFN2Bias", {ffn2_bias->Name()});
    op->SetInput("LnScale", {ln_scale->Name()});
    op->SetInput("LnBias", {ln_bias->Name()});
    op->SetInput("FFNLnScale", {ffn_ln_scale->Name()});
    op->SetInput("FFNLnBias", {ffn_ln_bias->Name()});
    op->SetAttr("pre_layer_norm", true);
    op->SetAttr("is_test", true);
    op->SetAttr("dropout_implementation", "upscale_in_train");
    op->SetAttr("dropout_rate", dropout_rate);
    op->SetAttr("epsilon", epsilon);
    op->SetOutput("Out", {out->Name()});
    op->SetOutput("CacheKVOut", {cache_kv_out->Name()});

    if (time_stamp) {
      op->SetInput("TimeStep", {time_stamp->Name()});
    }

    if (qkv_out_scale) {
      op->SetInput("QKVOutScale", {qkv_out_scale->Name()});
      op->SetInput("OutLinearOutScale", {out_linear_out_scale->Name()});
      op->SetInput("FFN1OutScale", {ffn1_out_scale->Name()});
      op->SetInput("FFN2OutScale", {ffn2_out_scale->Name()});
      op->SetAttr("qkv_in_scale", qkv_in_scale);
      op->SetAttr("out_linear_in_scale", out_linear_in_scale);
      op->SetAttr("ffn1_in_scale", ffn1_in_scale);
      op->SetAttr("ffn2_in_scale", ffn2_in_scale);
    }
    std::vector<VarDesc*> outs = {out, cache_kv_out};
    return outs;
  }

  VarDesc* dequantize_linear(VarDesc* x,
                             VarDesc* scale,
                             VarDesc* zero_point,
                             int bit_length = 8,
                             int quant_axis = -1) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("dequantize_linear");
    op->SetInput("X", {x->Name()});
    op->SetInput("Scale", {scale->Name()});
    op->SetInput("ZeroPoint", {zero_point->Name()});
    op->SetAttr("bit_length", bit_length);
    op->SetAttr("quant_axis", quant_axis);
    op->SetOutput("Y", {out->Name()});
    return out;
  }

  void backward(std::vector<VarDesc*> targets) {
    // This function is designed to simulate the structure of training program,
    //  but is constructed differently as the actual program.
    BlockDesc* block = program_.MutableBlock(0);
    std::vector<OpDesc*> forward_ops = block->AllOps();
    for (auto* var : targets) {
      OpDesc* none_op = block->AppendOp();
      none_op->SetType("none");
      none_op->SetInput("X", {var->Name()});
      VarDesc* grad_var =
          dense_tensor(GradVarName(var->Name()), var->GetShape(), false);
      none_op->SetOutput("Out", {grad_var->Name()});
    }
    for (int i = forward_ops.size() - 1; i >= 0; --i) {
      OpDesc* op = forward_ops[i];
      OpDesc* grad_op = block->AppendOp();
      grad_op->SetType(op->Type() + "_grad");
      // All op's inputs are grad_op's input.
      for (auto name : op->InputNames()) {
        grad_op->SetInput(name, op->Input(name));
      }
      // All op's outputs are grad_op's input.
      for (auto name : op->OutputNames()) {
        grad_op->SetInput(name, op->Output(name));
      }
      // All op's outputs grad are grad_op's input.
      for (auto name : op->OutputNames()) {
        std::vector<std::string> grad_var_names;
        for (auto var_name : op->Output(name)) {
          VarDesc* var = block->FindVar(var_name);
          VarDesc* grad_var =
              dense_tensor(GradVarName(var_name), var->GetShape(), false);
          grad_var_names.push_back(grad_var->Name());
        }
        grad_op->SetInput(GradVarName(name), grad_var_names);
      }
      // All op's inputs grad are grad_op's output.
      for (auto name : op->InputNames()) {
        std::vector<std::string> grad_var_names;
        for (auto var_name : op->Input(name)) {
          VarDesc* var = block->FindVar(var_name);
          VarDesc* grad_var =
              dense_tensor(GradVarName(var_name), var->GetShape(), false);
          grad_var_names.push_back(grad_var->Name());
        }
        grad_op->SetOutput(GradVarName(name), grad_var_names);
      }
      // TODO(liuyiqun): attrs
    }
  }

  VarDesc* cast(VarDesc* input, int in_dtype = 5, int out_dtype = 5) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("cast");
    op->SetInput("X", {input->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("in_dtype", in_dtype);
    op->SetAttr("out_dtype", out_dtype);
    return out;
  }

  VarDesc* range(VarDesc* start, VarDesc* end, VarDesc* step) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("range");
    op->SetInput("Start", {start->Name()});
    op->SetInput("End", {end->Name()});
    op->SetInput("Step", {step->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* flatten_contiguous_range(VarDesc* input) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("flatten_contiguous_range");
    op->SetInput("X", {input->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  std::vector<VarDesc*> beam_search(VarDesc* ids,
                                    VarDesc* scores,
                                    VarDesc* pre_ids,
                                    VarDesc* pre_scores,
                                    int beam_size = 1) {
    VarDesc* parent_idx = dense_tensor(unique_name());
    VarDesc* selected_ids = dense_tensor(unique_name());
    VarDesc* selected_scores = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("beam_search");
    op->SetInput("ids", {ids->Name()});
    op->SetInput("scores", {scores->Name()});
    op->SetInput("pre_ids", {pre_ids->Name()});
    op->SetInput("pre_scores", {pre_scores->Name()});
    op->SetOutput("parent_idx", {parent_idx->Name()});
    op->SetOutput("selected_ids", {selected_ids->Name()});
    op->SetOutput("selected_scores", {selected_scores->Name()});
    op->SetAttr("beam_size", 1);
    return {parent_idx, selected_ids, selected_scores};
  }

  VarDesc* lod_reset(VarDesc* x, VarDesc* y) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("lod_reset");
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* write_to_array(VarDesc* x, VarDesc* i) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("write_to_array");
    op->SetInput("X", {x->Name()});
    op->SetInput("I", {i->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* read_from_array(VarDesc* x, VarDesc* i) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("read_from_array");
    op->SetInput("X", {x->Name()});
    op->SetInput("I", {i->Name()});
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* gather(VarDesc* x, VarDesc* index, int axis) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("gather");
    op->SetInput("X", {x->Name()});
    op->SetInput("Index", {index->Name()});
    op->SetAttr("axis", axis);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* is_empty(VarDesc* input) { return unary_op("is_empty", input); }

  VarDesc* logical_not(VarDesc* input) {
    return unary_op("logical_not", input);
  }

  VarDesc* not_equal(VarDesc* x, VarDesc* y, int axis = -1) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("not_equal");
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetAttr("axis", axis);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

  VarDesc* stack(std::vector<VarDesc*> inputs, int axis = -1) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("stack");
    std::vector<std::string> input_names;
    for (auto* input : inputs) {
      input_names.push_back(input->Name());
    }
    op->SetInput("X", input_names);
    op->SetAttr("axis", axis);
    op->SetOutput("Y", {out->Name()});
    return out;
  }

  VarDesc* tile(VarDesc* x, const std::vector<int>& repeat_times = {2}) {
    VarDesc* out = dense_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("tile");
    op->SetInput("X", {x->Name()});
    op->SetAttr("repeat_times", repeat_times);
    op->SetOutput("Out", {out->Name()});
    return out;
  }

 private:
  VarDesc* dense_tensor(std::string name,
                        std::vector<int64_t> shape = {},
                        bool is_persistable = false,
                        proto::VarType::Type data_type = proto::VarType::FP32) {
    auto* var = program_.MutableBlock(0)->Var(name);
    var->SetType(proto::VarType::DENSE_TENSOR);
    var->SetDataType(data_type);
    var->SetShape(shape);
    var->SetPersistable(is_persistable);
    return var;
  }

  VarDesc* unary_op(std::string type,
                    VarDesc* x,
                    VarDesc* out = nullptr,
                    const AttributeMap* attrs = nullptr) {
    if (!out) {
      out = dense_tensor(unique_name());
    }
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    if (attrs) {
      for (auto& iter : *attrs) {
        op->SetAttr(iter.first, iter.second);
      }
    }
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* binary_op(std::string type,
                     VarDesc* x,
                     VarDesc* y,
                     VarDesc* out = nullptr,
                     const AttributeMap* attrs = nullptr) {
    if (!out) {
      out = dense_tensor(unique_name());
    }
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetOutput("Out", {out->Name()});
    if (attrs) {
      for (auto& iter : *attrs) {
        op->SetAttr(iter.first, iter.second);
      }
    }
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  std::string unique_name() { return "tmp_" + std::to_string(idx_++); }

 private:
  ProgramDesc program_;
  int idx_{0};
};

static std::string DebugString(OpDesc* op) {
  std::ostringstream os;
  os << "Op(" << op->Type() << "), inputs:{";
  bool is_first = true;
  for (auto& name : op->InputNames()) {
    if (!is_first) {
      os << ", ";
    }
    os << name << "[";
    bool is_first_var_name = true;
    for (auto& var_name : op->Input(name)) {
      if (!is_first_var_name) {
        os << ", ";
      }
      os << var_name;
      is_first_var_name = false;
    }
    os << "]";
    is_first = false;
  }

  os << "}, outputs:{";
  is_first = true;
  for (auto& name : op->OutputNames()) {
    if (!is_first) {
      os << ", ";
    }
    os << name << "[";
    bool is_first_var_name = true;
    for (auto& var_name : op->Output(name)) {
      if (!is_first_var_name) {
        os << ", ";
      }
      os << var_name;
      is_first_var_name = false;
    }
    os << "]";
    is_first = false;
  }
  os << "}";
  return os.str();
}

static std::string DebugString(const Node* node) {
  std::ostringstream os;
  if (node->IsOp() && node->Op()) {
    OpDesc* op = node->Op();
    os << "Node(" << DebugString(op) << "), inputs:{";
    bool is_first = true;
    for (auto* in : node->inputs) {
      if (!is_first) {
        os << ", ";
      }
      os << in->Name();
      is_first = false;
    }
    os << "}, outputs:{";
    is_first = true;
    for (auto* out : node->outputs) {
      if (!is_first) {
        os << ", ";
      }
      os << out->Name();
      is_first = false;
    }
    os << "}.";
  } else {
    os << "Node(" << node->Name();
    if (node->IsVar() && node->Var()) {
      os << "{";
      bool is_first = true;
      for (auto dim : node->Var()->GetShape()) {
        if (!is_first) {
          os << "x";
        }
        os << dim;
        is_first = false;
      }
      os << "}";
    }
    os << "), inputs:{";
    bool is_first = true;
    for (auto* in : node->inputs) {
      if (!is_first) {
        os << ", ";
      }
      if (in->IsOp() && in->Op()) {
        os << in->Op()->Type();
      }
      is_first = false;
    }
    os << "}, outputs:{";
    is_first = true;
    for (auto* out : node->outputs) {
      if (!is_first) {
        os << ", ";
      }
      if (out->IsOp() && out->Op()) {
        os << out->Op()->Type();
      }
      is_first = false;
    }
    os << "}";
  }
  return os.str();
}

static std::string DebugString(const std::vector<Node*>& nodes) {
  std::ostringstream os;
  for (auto* node : nodes) {
    if (node->IsOp() && node->Op()) {
      os << "  ";
    } else if ((node->IsVar() && node->Var()) || node->IsCtrlVar()) {
      os << "    ";
    }
    os << DebugString(node) << "\n";
  }
  return os.str();
}

static std::string DebugString(const std::unordered_set<Node*>& nodes) {
  std::vector<Node*> vec;
  for (auto* node : nodes) {
    vec.push_back(node);
  }
  return DebugString(vec);
}

static std::string DebugString(Graph* graph) {
  std::ostringstream os;
  os << "Graph: {\n" << DebugString(graph->Nodes()) << "}\n";
  return os.str();
}

static std::string DebugString(const std::unique_ptr<Graph>& graph) {
  return DebugString(graph.get());
}

static std::vector<ir::Node*> GetOpNodes(const std::unique_ptr<Graph>& graph,
                                         std::string op_type) {
  std::vector<ir::Node*> rc;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() && node->Op()->Type() == op_type) {
      rc.push_back(node);
    }
  }
  return rc;
}

static int GetNumOpNodes(const std::unique_ptr<Graph>& graph,
                         std::string op_type = "") {
  int num_nodes = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() &&
        (node->Op()->Type() == op_type || op_type.empty())) {
      num_nodes++;
    }
  }
  return num_nodes;
}

static void RegisterOpKernel(std::vector<std::string>&& op_types) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();

  phi::CPUPlace place = phi::CPUPlace();
  OpKernelType mkldnn_kernel_type = OpKernelType(proto::VarType::FP32,
                                                 place,
                                                 DataLayout::kAnyLayout,
                                                 LibraryType::kMKLDNN);

  auto fake_kernel_func = [](const ExecutionContext&) -> void {};

  for (auto& op_name : op_types)
    all_kernels[op_name][mkldnn_kernel_type] = fake_kernel_func;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
