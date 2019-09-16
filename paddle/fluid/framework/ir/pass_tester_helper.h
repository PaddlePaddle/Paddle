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
#include <vector>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

struct Layers {
 public:
  const ProgramDesc& main_program() { return program_; }

  VarDesc* data(std::string name, std::vector<int64_t> shape = {},
                bool is_persistable = false) {
    return lod_tensor(name, shape, is_persistable);
  }

  VarDesc* conv2d(VarDesc* input, VarDesc* filter, VarDesc* bias,
                  bool use_cudnn = false) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("conv2d");
    op->SetInput("Input", {input->Name()});
    op->SetInput("Filter", {filter->Name()});
    op->SetInput("Bias", {bias->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* depthwise_conv2d(VarDesc* input, VarDesc* filter, VarDesc* bias,
                            bool use_cudnn) {
    VarDesc* out = lod_tensor(unique_name());
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

  VarDesc* pool2d(VarDesc* x, bool use_cudnn) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("pool2d");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("use_cudnn", use_cudnn);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* relu(VarDesc* x, VarDesc* out = nullptr) {
    return unary_op("relu", x, out);
  }

  VarDesc* fc(VarDesc* input, VarDesc* w, VarDesc* bias,
              int in_num_col_dims = 1, std::string activation_type = "") {
    VarDesc* out = lod_tensor(unique_name());
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

  VarDesc* mul(VarDesc* x, VarDesc* y, VarDesc* out = nullptr,
               int x_num_col_dims = 1) {
    AttributeMap attrs;
    attrs["x_num_col_dims"] = 1;
    return binary_op("mul", x, y, out, &attrs);
  }

  VarDesc* elementwise_add(VarDesc* x, VarDesc* y, VarDesc* out = nullptr) {
    return binary_op("elementwise_add", x, y, out);
  }

  VarDesc* dropout(VarDesc* x, float dropout_prob,
                   std::string dropout_implementation) {
    VarDesc* out = lod_tensor(unique_name());
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
    VarDesc* out = lod_tensor(unique_name());
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

  std::vector<VarDesc*> layer_norm(VarDesc* x, VarDesc* scale = nullptr,
                                   VarDesc* bias = nullptr) {
    VarDesc* y = lod_tensor(unique_name());
    VarDesc* mean = lod_tensor(unique_name());
    VarDesc* variance = lod_tensor(unique_name());
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

 private:
  VarDesc* lod_tensor(std::string name, std::vector<int64_t> shape = {},
                      bool is_persistable = false) {
    auto* var = program_.MutableBlock(0)->Var(name);
    var->SetType(proto::VarType::LOD_TENSOR);
    var->SetShape(shape);
    var->SetPersistable(is_persistable);
    return var;
  }

  VarDesc* unary_op(std::string type, VarDesc* x, VarDesc* out = nullptr) {
    if (!out) {
      out = lod_tensor(unique_name());
    }
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  VarDesc* binary_op(std::string type, VarDesc* x, VarDesc* y,
                     VarDesc* out = nullptr,
                     const AttributeMap* attrs = nullptr) {
    if (!out) {
      out = lod_tensor(unique_name());
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

static std::string DebugString(Node* node) {
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
  } else if (node->IsVar() && node->Var()) {
    os << "Node(" << node->Name() << "), inputs:{";
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

static std::string DebugString(const std::unique_ptr<Graph>& graph) {
  std::ostringstream os;
  os << "Graph: {\n";
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()) {
      os << "  ";
    } else if (node->IsVar() && node->Var()) {
      os << "    ";
    }
    os << DebugString(node) << "\n";
  }
  os << "}\n";
  return os.str();
}

static int GetNumOpNodes(const std::unique_ptr<Graph>& graph,
                         std::string op_type) {
  int num_nodes = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() && node->Op()->Type() == op_type) {
      num_nodes++;
    }
  }
  return num_nodes;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
