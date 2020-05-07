// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fc_parallel_fuse_pass.h"

#include <string>
#include <tuple>
#include <vector>

#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T>
bool GetAttrFromThreeOps(std::string attr_name, const ir::Node* op1,
                         const ir::Node* op2, const ir::Node* op3,
                         T* out_attr_val) {
  *out_attr_val = op1->Op()->GetAttrIfExists<T>(attr_name);
  if (*out_attr_val != op2->Op()->GetAttrIfExists<T>(attr_name) ||
      *out_attr_val != op3->Op()->GetAttrIfExists<T>(attr_name))
    return false;
  return true;
}

template <typename T>
bool CopyAttrIfConsistent(std::string attr_name, const ir::Node* op1,
                          const ir::Node* op2, const ir::Node* op3,
                          OpDesc* fc_new_desc) {
  T attr;
  if (!GetAttrFromThreeOps(attr_name, op1, op2, op3, &attr)) {
    string::PrettyLogDetail("%s is not consistent. Not fusing.", attr_name);
    return false;
  }
  fc_new_desc->SetAttr(attr_name, attr);
  return true;
}

void ConcatWeights(const LoDTensor& w1, const LoDTensor& w2,
                   const LoDTensor& w3, LoDTensor* fc_new_weights_tensor,
                   bool padding_weights) {
  if (padding_weights) {
    float* new_data =
        fc_new_weights_tensor->mutable_data<float>(platform::CPUPlace());
    auto new_width = fc_new_weights_tensor->dims()[1];
    auto width = w1.dims()[1] - 4;
    auto stride = w1.dims()[1];
    for (int row = 0; row < fc_new_weights_tensor->dims()[0]; ++row) {
      memcpy(new_data + 0 + row * new_width, w1.data<float>() + row * stride,
             width * sizeof(float));
      memcpy(new_data + width + row * new_width,
             w2.data<float>() + row * stride, width * sizeof(float));
      // copy w3 with padding width
      memcpy(new_data + 2 * width + row * new_width,
             w3.data<float>() + row * stride, w3.dims()[1] * sizeof(float));
    }
  } else {
    using EMAM = Eigen::Map<
        Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using CEMAM =
        Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>>;
    CEMAM w1_a(w1.data<float>(), w1.dims()[0], w1.dims()[1]);
    CEMAM w2_a(w2.data<float>(), w2.dims()[0], w2.dims()[1]);
    CEMAM w3_a(w3.data<float>(), w3.dims()[0], w3.dims()[1]);
    EMAM result_a(
        fc_new_weights_tensor->mutable_data<float>(platform::CPUPlace()),
        fc_new_weights_tensor->dims()[0], fc_new_weights_tensor->dims()[1]);
    result_a << w1_a, w2_a, w3_a;
  }
}

template <typename T>
void ConcatBiases(const LoDTensor& b1, const LoDTensor& b2, const LoDTensor& b3,
                  LoDTensor* fc_new_bias_tensor) {
  using EVAM = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
  using CEVAM = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

  CEVAM b1_a(b1.data<T>(), b1.numel(), 1);
  CEVAM b2_a(b2.data<T>(), b2.numel(), 1);
  CEVAM b3_a(b3.data<T>(), b3.numel(), 1);
  EVAM result_a(fc_new_bias_tensor->mutable_data<T>(platform::CPUPlace()),
                fc_new_bias_tensor->numel(), 1);
  result_a << b1_a, b2_a, b3_a;
}

void ConcatBiases(const LoDTensor& b1, const LoDTensor& b2, const LoDTensor& b3,
                  LoDTensor* fc_new_bias_tensor) {
  if (b1.type() == proto::VarType::FP32) {
    ConcatBiases<float>(b1, b2, b3, fc_new_bias_tensor);
  } else if (b1.type() == proto::VarType::UINT8) {
    ConcatBiases<uint8_t>(b1, b2, b3, fc_new_bias_tensor);
  } else {
    throw platform::errors::Unimplemented(
        "Unexpected data type found %s in FC biases during "
        "fc_parallel_fuse_pass.",
        DataTypeToString(b1.type()));
  }
}

void FcParallelFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(scope, platform::errors::PreconditionNotMet(
                                     "The scope should not be nullptr."));

  GraphPatternDetector gpd;
  patterns::FcParallel fc_parallel_pattern(gpd.mutable_pattern(), name_scope_);
  fc_parallel_pattern();

  int found_fc_parallel_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FcParallel fuse";

    GET_IR_NODE_FROM_SUBGRAPH(fc_in, fc_in, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc1, fc1, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc2, fc2, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc3, fc3, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc1_w, fc1_w, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc2_w, fc2_w, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc3_w, fc3_w, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc1_b, fc1_b, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc2_b, fc2_b, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc3_b, fc3_b, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc1_out, fc1_out, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc2_out, fc2_out, fc_parallel_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc3_out, fc3_out, fc_parallel_pattern);

    // Create new FC node that replaces the 3 fused ones.
    OpDesc fc_new_desc;
    fc_new_desc.SetType("fc");

    if (!CopyAttrIfConsistent<int>("in_num_col_dims", fc1, fc2, fc3,
                                   &fc_new_desc))
      return;

    if (!CopyAttrIfConsistent<float>("Scale_in", fc1, fc2, fc3, &fc_new_desc))
      return;

    if (!CopyAttrIfConsistent<float>("Scale_out", fc1, fc2, fc3, &fc_new_desc))
      return;

    if (!CopyAttrIfConsistent<std::string>("activation_type", fc1, fc2, fc3,
                                           &fc_new_desc))
      return;

    if (!CopyAttrIfConsistent<bool>("use_mkldnn", fc1, fc2, fc3, &fc_new_desc))
      return;

    if (!CopyAttrIfConsistent<bool>("padding_weights", fc1, fc2, fc3,
                                    &fc_new_desc))
      return;
    auto padding_weights = fc_new_desc.GetAttrIfExists<bool>("padding_weights");

    // Get weights tensors
    auto* w1 = scope->FindVar(fc1_w->Name())->GetMutable<LoDTensor>();
    auto* w2 = scope->FindVar(fc2_w->Name())->GetMutable<LoDTensor>();
    auto* w3 = scope->FindVar(fc3_w->Name())->GetMutable<LoDTensor>();

    // Infer combined weights shape
    if (w1->dims() != w2->dims() || w1->dims() != w3->dims()) return;
    framework::DDim fc_new_weights_dims = w1->dims();
    fc_new_weights_dims[1] += w2->dims()[1] + w3->dims()[1];
    if (padding_weights) {
      fc_new_weights_dims[1] -= 4 * 2;
    }

    // Create combined weights variable
    VarDesc fc_new_weights_desc(
        patterns::PDNodeName(name_scope_, "fc_new_weights"));
    fc_new_weights_desc.SetShape(framework::vectorize(fc_new_weights_dims));
    fc_new_weights_desc.SetDataType(w1->type());
    fc_new_weights_desc.SetLoDLevel(0);
    fc_new_weights_desc.SetPersistable(true);
    ir::Node* fc_new_weights = graph->CreateVarNode(&fc_new_weights_desc);

    // fill weights with data
    LoDTensor* fc_new_weights_tensor =
        scope->Var(fc_new_weights->Name())->GetMutable<LoDTensor>();
    fc_new_weights_tensor->Resize(fc_new_weights_dims);
    ConcatWeights(*w1, *w2, *w3, fc_new_weights_tensor, padding_weights);

    // Get bias tensors
    auto* b1 = scope->FindVar(fc1_b->Name())->GetMutable<LoDTensor>();
    auto* b2 = scope->FindVar(fc2_b->Name())->GetMutable<LoDTensor>();
    auto* b3 = scope->FindVar(fc3_b->Name())->GetMutable<LoDTensor>();

    // Infer combined biases shape
    if (b1->dims() != b2->dims() || b1->dims() != b3->dims()) return;
    framework::DDim fc_new_bias_dims = b1->dims();
    fc_new_bias_dims[0] += b2->dims()[0] + b3->dims()[0];

    // Create combined biases variable
    VarDesc fc_new_bias_desc(patterns::PDNodeName(name_scope_, "fc_new_bias"));
    fc_new_bias_desc.SetShape(framework::vectorize(fc_new_bias_dims));
    fc_new_bias_desc.SetDataType(b1->type());
    fc_new_bias_desc.SetLoDLevel(0);
    fc_new_bias_desc.SetPersistable(true);
    ir::Node* fc_new_bias = graph->CreateVarNode(&fc_new_bias_desc);

    // fill bias with data
    LoDTensor* fc_new_bias_tensor =
        scope->Var(fc_new_bias->Name())->GetMutable<LoDTensor>();
    fc_new_bias_tensor->Resize(fc_new_bias_dims);
    ConcatBiases(*b1, *b2, *b3, fc_new_bias_tensor);

    // Create FC output variable
    VarDesc fc_new_out_desc(patterns::PDNodeName(name_scope_, "fc_new_out"));
    fc_new_out_desc.SetPersistable(false);
    auto* fc_new_out = graph->CreateVarNode(&fc_new_out_desc);

    // Create Split op
    OpDesc split_desc;
    split_desc.SetInput("X", {fc_new_out->Name()});
    split_desc.SetOutput("Out",
                         {fc1_out->Name(), fc2_out->Name(), fc3_out->Name()});
    split_desc.SetType("split");
    split_desc.SetAttr("axis", 2);
    split_desc.SetAttr("num", 3);
    auto split_op = graph->CreateOpNode(&split_desc);

    // Connect nodes to the new FC op
    fc_new_desc.SetInput("Input", {fc_in->Name()});
    fc_new_desc.SetInput("W", {fc_new_weights->Name()});
    fc_new_desc.SetInput("Bias", {fc_new_bias->Name()});
    fc_new_desc.SetOutput("Out", {fc_new_out->Name()});
    ir::Node* fc_new = graph->CreateOpNode(&fc_new_desc);

    GraphSafeRemoveNodes(
        graph, {fc1, fc1_w, fc1_b, fc2, fc2_w, fc2_b, fc3, fc3_w, fc3_b});

    IR_NODE_LINK_TO(fc_in, fc_new);
    IR_NODE_LINK_TO(fc_new_weights, fc_new);
    IR_NODE_LINK_TO(fc_new_bias, fc_new);
    IR_NODE_LINK_TO(fc_new, fc_new_out);
    IR_NODE_LINK_TO(fc_new_out, split_op);
    IR_NODE_LINK_TO(split_op, fc1_out);
    IR_NODE_LINK_TO(split_op, fc2_out);
    IR_NODE_LINK_TO(split_op, fc3_out);

    ++found_fc_parallel_count;
  };

  gpd(graph, handler);
  AddStatis(found_fc_parallel_count);

  std::stringstream msg_ss;
  msg_ss << "---    Fused " << found_fc_parallel_count
         << " FcParallel patterns";
  string::PrettyLogDetail(msg_ss.str().c_str());
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_parallel_fuse_pass, paddle::framework::ir::FcParallelFusePass);
