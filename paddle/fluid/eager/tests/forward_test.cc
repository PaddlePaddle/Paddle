// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/nodes/scale_node.h"

#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/tensor_meta.h"

#include "paddle/fluid/eager/tests/test_utils.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(Forward, SingleNode) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  pt::Tensor t = EagerUtils::CreateTensorWithValue(
      ddim, pt::Backend::kCPU, pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
      5.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(t));
  pt::Tensor& tensor = target_tensors[0];
  EagerUtils::autograd_meta(&tensor)->SetStopGradient(false);

  // Run Forward
  float scale = 2.0;
  float bias = 3.0;
  pt::Tensor out = egr::scale(tensor, scale, bias, true /*bias_after_scale*/,
                              true /*trace_backward*/);

  // Examine Forward Output
  CompareTensorWithValue<float>(out, 13.0);

  // Examine GradNode
  {
    // 1. GradNode
    AutogradMeta* meta = EagerUtils::autograd_meta(&out);
    GradNodeBase* grad_node = meta->GradNode();
    GradNodeScale* scale_node = dynamic_cast<GradNodeScale*>(grad_node);
    PADDLE_ENFORCE(
        scale_node != nullptr,
        paddle::platform::errors::Fatal("AutogradMeta should have held "
                                        "grad_node with type GradNodeScale*"));
    PADDLE_ENFORCE(
        (meta->OutRankInfo().first == 0) && (meta->OutRankInfo().second == 0),
        paddle::platform::errors::Fatal(
            "OutRankInfo in AutogradMeta should have been 0"));

    // 2. TensorWrapper: No TensorWrapper for ScaleNode
    // 3. NextEdges: No NextEdges for Single Node Case
  }
}

/*
 inp
  |
Node0
  |
Node1
  |
 out
*/
TEST(Forward, LinearNodes) {
  InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  pt::Tensor t = EagerUtils::CreateTensorWithValue(
      ddim, pt::Backend::kCPU, pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
      5.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(t));
  pt::Tensor& tensor = target_tensors[0];
  EagerUtils::autograd_meta(&tensor)->SetStopGradient(false);

  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  pt::Tensor out0 = egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/,
                               true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  pt::Tensor out1 = egr::scale(out0, scale1, bias1, true /*bias_after_scale*/,
                               true /*trace_backward*/);

  // Examine Forward Output 0
  CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  CompareTensorWithValue<float>(out1, 75.0);

  // Examine GradNode
  {
    // 1. GradNode
    // Node 0
    AutogradMeta* meta0 = EagerUtils::autograd_meta(&out0);
    GradNodeBase* grad_node0 = meta0->GradNode();
    GradNodeScale* scale_node0 = dynamic_cast<GradNodeScale*>(grad_node0);
    PADDLE_ENFORCE(
        scale_node0 != nullptr,
        paddle::platform::errors::Fatal("AutogradMeta should have held "
                                        "grad_node with type GradNodeScale*"));
    PADDLE_ENFORCE(
        (meta0->OutRankInfo().first == 0) && (meta0->OutRankInfo().second == 0),
        paddle::platform::errors::Fatal(
            "OutRankInfo in AutogradMeta should have been 0"));

    // Node 1
    AutogradMeta* meta1 = EagerUtils::autograd_meta(&out1);
    GradNodeBase* grad_node1 = meta1->GradNode();
    GradNodeScale* scale_node1 = dynamic_cast<GradNodeScale*>(grad_node1);
    PADDLE_ENFORCE(
        scale_node1 != nullptr,
        paddle::platform::errors::Fatal("AutogradMeta should have held "
                                        "grad_node with type GradNodeScale*"));
    PADDLE_ENFORCE(
        (meta1->OutRankInfo().first == 0) && (meta1->OutRankInfo().second == 0),
        paddle::platform::errors::Fatal(
            "OutRankInfo in AutogradMeta should have been 0"));

    // 2. TensorWrapper: No TensorWrapper for ScaleNode
    // 3. NextEdges: Node 1 -> Node 0
    const std::vector<std::vector<Edge>>& node1_edges = grad_node1->GetEdges();
    PADDLE_ENFORCE(node1_edges.size() == 1,
                   paddle::platform::errors::Fatal(
                       "Node 1 should have exactly 1 edge slot "));

    const auto& node1_edge = node1_edges[0];
    PADDLE_ENFORCE(node1_edge.size() == 1,
                   paddle::platform::errors::Fatal(
                       "Node 1 should have exactly 1 edge in slot 0"));
    PADDLE_ENFORCE((node1_edge[0].GetEdgeRankInfo().first == 0) &&
                       (node1_edge[0].GetEdgeRankInfo().second == 0),
                   paddle::platform::errors::Fatal(
                       "Node1's edge should have input rank of 0"));

    PADDLE_ENFORCE(
        node1_edge[0].GetGradNode() == grad_node0,
        paddle::platform::errors::Fatal("Node1's edge should point to Node 0"));
  }
}

/*
       inp
        |
      Node0
    ____|____
    |       |
  Node1   Node2
    |       |
   out1    out2
*/
TEST(Forward, BranchedNodes) {
  InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  pt::Tensor t = EagerUtils::CreateTensorWithValue(
      ddim, pt::Backend::kCPU, pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
      5.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(t));
  pt::Tensor& tensor = target_tensors[0];
  EagerUtils::autograd_meta(&tensor)->SetStopGradient(false);

  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  pt::Tensor out0 = egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/,
                               true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  pt::Tensor out1 = egr::scale(out0, scale1, bias1, true /*bias_after_scale*/,
                               true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  pt::Tensor out2 = egr::scale(out0, scale2, bias2, true /*bias_after_scale*/,
                               true /*trace_backward*/);

  // Examine Forward Output 0
  CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  CompareTensorWithValue<float>(out1, 75.0);

  // Examine Forward Output 2
  CompareTensorWithValue<float>(out2, 150.0);

  // Examine GradNode
  {
    // 1. GradNode
    // Node 0
    AutogradMeta* meta0 = EagerUtils::autograd_meta(&out0);
    GradNodeBase* grad_node0 = meta0->GradNode();
    GradNodeScale* scale_node0 = dynamic_cast<GradNodeScale*>(grad_node0);
    PADDLE_ENFORCE(
        scale_node0 != nullptr,
        paddle::platform::errors::Fatal("AutogradMeta should have held "
                                        "grad_node with type GradNodeScale*"));
    PADDLE_ENFORCE(
        (meta0->OutRankInfo().first == 0) && (meta0->OutRankInfo().second == 0),
        paddle::platform::errors::Fatal(
            "OutRankInfo in AutogradMeta should have been 0"));

    // Node 1
    AutogradMeta* meta1 = EagerUtils::autograd_meta(&out1);
    GradNodeBase* grad_node1 = meta1->GradNode();
    GradNodeScale* scale_node1 = dynamic_cast<GradNodeScale*>(grad_node1);
    PADDLE_ENFORCE(
        scale_node1 != nullptr,
        paddle::platform::errors::Fatal("AutogradMeta should have held "
                                        "grad_node with type GradNodeScale*"));
    PADDLE_ENFORCE(
        (meta1->OutRankInfo().first == 0) && (meta1->OutRankInfo().second == 0),
        paddle::platform::errors::Fatal(
            "OutRankInfo in AutogradMeta should have been 0"));

    // Node 2
    AutogradMeta* meta2 = EagerUtils::autograd_meta(&out2);
    GradNodeBase* grad_node2 = meta2->GradNode();
    GradNodeScale* scale_node2 = dynamic_cast<GradNodeScale*>(grad_node2);
    PADDLE_ENFORCE(
        scale_node2 != nullptr,
        paddle::platform::errors::Fatal("AutogradMeta should have held "
                                        "grad_node with type GradNodeScale*"));
    PADDLE_ENFORCE(
        (meta2->OutRankInfo().first == 0) && (meta2->OutRankInfo().second == 0),
        paddle::platform::errors::Fatal(
            "OutRankInfo in AutogradMeta should have been 0"));

    // 2. TensorWrapper: No TensorWrapper for ScaleNode
    // 3. NextEdges
    // Node 1 -> Node 0
    const std::vector<std::vector<Edge>>& node1_edges = grad_node1->GetEdges();
    PADDLE_ENFORCE(node1_edges.size() == 1,
                   paddle::platform::errors::Fatal(
                       "Node 1 should have exactly 1 edge slot"));
    PADDLE_ENFORCE(node1_edges[0].size() == 1,
                   paddle::platform::errors::Fatal(
                       "Node 1 should have exactly 1 edge in slot 0"));
    const Edge& node1_edge = node1_edges[0][0];
    PADDLE_ENFORCE((node1_edge.GetEdgeRankInfo().first == 0) &&
                       (node1_edge.GetEdgeRankInfo().second == 0),
                   paddle::platform::errors::Fatal(
                       "Node1's edge should have input rank of 0"));
    PADDLE_ENFORCE(
        node1_edge.GetGradNode() == grad_node0,
        paddle::platform::errors::Fatal("Node1's edge should point to Node 0"));

    // Node 2 -> Node 0
    const std::vector<std::vector<Edge>>& node2_edges = grad_node2->GetEdges();
    PADDLE_ENFORCE(node2_edges.size() == 1,
                   paddle::platform::errors::Fatal(
                       "Node 2 should have exactly 1 edge slot"));
    PADDLE_ENFORCE(node1_edges[0].size() == 1,
                   paddle::platform::errors::Fatal(
                       "Node 2 should have exactly 1 edge in slot 0"));
    const Edge& node2_edge = node2_edges[0][0];
    PADDLE_ENFORCE((node2_edge.GetEdgeRankInfo().first == 0) &&
                       (node2_edge.GetEdgeRankInfo().second == 0),
                   paddle::platform::errors::Fatal(
                       "Node2's edge should have input rank of 0"));
    PADDLE_ENFORCE(
        node2_edge.GetGradNode() == grad_node0,
        paddle::platform::errors::Fatal("Node2's edge should point to Node 0"));
  }
}
