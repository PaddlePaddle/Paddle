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
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "test/cpp/eager/test_utils.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace egr {

TEST(Forward, SingleNode) {
  // Prepare Device Contexts
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::Tensor> target_tensors;
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor t = eager_test::CreateTensorWithValue(ddim,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       5.0 /*value*/,
                                                       false /*is_leaf*/);
  target_tensors.emplace_back(std::move(t));
  paddle::Tensor& tensor = target_tensors[0];
  EagerUtils::autograd_meta(&tensor)->SetStopGradient(false);

  // Run Forward
  float scale = 2.0;
  float bias = 3.0;
  paddle::Tensor out = egr::scale(
      tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output
  eager_test::CompareTensorWithValue<float>(out, 13.0);

  // Examine GradNode
  {
    // 1. GradNode
    AutogradMeta* meta = EagerUtils::autograd_meta(&out);
    GradNodeBase* grad_node = meta->GradNode();
    GradNodeScale* scale_node = dynamic_cast<GradNodeScale*>(grad_node);

    CHECK_NOTNULL(scale_node);
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta->OutRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta->OutRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta->OutRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta->OutRankInfo().second) is not 0"));
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
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::Tensor> target_tensors;
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor t = eager_test::CreateTensorWithValue(ddim,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       5.0 /*value*/,
                                                       false /*is_leaf*/);
  target_tensors.emplace_back(std::move(t));
  paddle::Tensor& tensor = target_tensors[0];
  EagerUtils::autograd_meta(&tensor)->SetStopGradient(false);

  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  paddle::Tensor out0 = egr::scale(tensor,
                                   scale0,
                                   bias0,
                                   true /*bias_after_scale*/,
                                   true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  paddle::Tensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  eager_test::CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  eager_test::CompareTensorWithValue<float>(out1, 75.0);

  // Examine GradNode
  {
    // 1. GradNode
    // Node 0
    AutogradMeta* meta0 = EagerUtils::autograd_meta(&out0);
    GradNodeBase* grad_node0 = meta0->GradNode();
    GradNodeScale* scale_node0 = dynamic_cast<GradNodeScale*>(grad_node0);

    CHECK_NOTNULL(scale_node0);
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta0->OutRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta0->OutRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta0->OutRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta0->OutRankInfo().second) is not 0"));

    // Node 1
    AutogradMeta* meta1 = EagerUtils::autograd_meta(&out1);
    GradNodeBase* grad_node1 = meta1->GradNode();
    GradNodeScale* scale_node1 = dynamic_cast<GradNodeScale*>(grad_node1);

    CHECK_NOTNULL(scale_node1);
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta1->OutRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta1->OutRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta1->OutRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta1->OutRankInfo().second) is not 0"));

    // 2. TensorWrapper: No TensorWrapper for ScaleNode
    // 3. NextEdges: Node 1 -> Node 0
    const paddle::small_vector<std::vector<GradSlotMeta>,
                               egr::kSlotSmallVectorSize>& node1_metas =
        grad_node1->OutputMeta();
    const auto& node1_meta = node1_metas[0];

    PADDLE_ENFORCE_EQ(
        static_cast<int>(node1_meta[0].GetEdge().GetEdgeRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(node1_meta[0].GetEdge().GetEdgeRankInfo().first)"
            "is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(node1_meta[0].GetEdge().GetEdgeRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(node1_meta[0].GetEdge().GetEdgeRankInfo().second)"
            "is not 0"));
    PADDLE_ENFORCE_EQ(node1_meta[0].GetEdge().GetGradNode(),
                      grad_node0,
                      common::errors::InvalidArgument(
                          "node1_meta[0].GetEdge().GetGradNode()"
                          "is not equal with grad_node0"
                          "the value of grad_node0 is %d"
                          "and node1_meta[0].GetEdge().GetGradNode() is %d",
                          grad_node0,
                          node1_meta[0].GetEdge().GetGradNode()));
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
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::Tensor> target_tensors;
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor t = eager_test::CreateTensorWithValue(ddim,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       5.0 /*value*/,
                                                       false /*is_leaf*/);
  target_tensors.emplace_back(std::move(t));
  paddle::Tensor& tensor = target_tensors[0];
  EagerUtils::autograd_meta(&tensor)->SetStopGradient(false);

  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  paddle::Tensor out0 = egr::scale(tensor,
                                   scale0,
                                   bias0,
                                   true /*bias_after_scale*/,
                                   true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  paddle::Tensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  paddle::Tensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  eager_test::CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  eager_test::CompareTensorWithValue<float>(out1, 75.0);

  // Examine Forward Output 2
  eager_test::CompareTensorWithValue<float>(out2, 150.0);

  // Examine GradNode
  {
    // 1. GradNode
    // Node 0
    AutogradMeta* meta0 = EagerUtils::autograd_meta(&out0);
    GradNodeBase* grad_node0 = meta0->GradNode();
    GradNodeScale* scale_node0 = dynamic_cast<GradNodeScale*>(grad_node0);

    CHECK_NOTNULL(scale_node0);
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta0->OutRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta0->OutRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta0->OutRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta0->OutRankInfo().second) is not 0"));

    // Node 1
    AutogradMeta* meta1 = EagerUtils::autograd_meta(&out1);
    GradNodeBase* grad_node1 = meta1->GradNode();
    GradNodeScale* scale_node1 = dynamic_cast<GradNodeScale*>(grad_node1);

    CHECK_NOTNULL(scale_node1);
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta1->OutRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta1->OutRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta1->OutRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta1->OutRankInfo().second) is not 0"));

    // Node 2
    AutogradMeta* meta2 = EagerUtils::autograd_meta(&out2);
    GradNodeBase* grad_node2 = meta2->GradNode();
    GradNodeScale* scale_node2 = dynamic_cast<GradNodeScale*>(grad_node2);

    CHECK_NOTNULL(scale_node2);
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta2->OutRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta2->OutRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(meta2->OutRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(meta2->OutRankInfo().second) is not 0"));

    // 2. TensorWrapper: No TensorWrapper for ScaleNode
    // 3. NextEdges
    // Node 1 -> Node 0
    const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
        node1_metas = grad_node1->OutputMeta();
    const Edge& node1_edge = node1_metas[0][0].GetEdge();

    PADDLE_ENFORCE_EQ(
        static_cast<int>(node1_edge.GetEdgeRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(node1_edge.GetEdgeRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(node1_edge.GetEdgeRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(node1_edge.GetEdgeRankInfo().second) is not 0"));
    PADDLE_ENFORCE_EQ(
        node1_edge.GetGradNode(),
        grad_node0,
        common::errors::InvalidArgument(
            "node1_edge.GetGradNode() is not equal with grad_node0"
            "the value of node1_edge.GetGradNode() is %d and grad_node0 is %d",
            node1_edge.GetGradNode(),
            grad_node0));

    // Node 2 -> Node 0
    const paddle::small_vector<std::vector<egr::GradSlotMeta>,
                               egr::kSlotSmallVectorSize>& node2_metas =
        grad_node2->OutputMeta();
    const Edge& node2_edge = node2_metas[0][0].GetEdge();

    PADDLE_ENFORCE_EQ(
        static_cast<int>(node2_edge.GetEdgeRankInfo().first),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(node2_edge.GetEdgeRankInfo().first) is not 0"));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(node2_edge.GetEdgeRankInfo().second),
        0,
        common::errors::InvalidArgument(
            "static_cast<int>(node2_edge.GetEdgeRankInfo().second) is not 0"));
    PADDLE_ENFORCE_EQ(
        node2_edge.GetGradNode(),
        grad_node0,
        common::errors::InvalidArgument(
            "node2_edge.GetGradNode() is not equal with grad_node0"
            "the value of node2_edge.GetGradNode() is %d and grad_node0 is %d",
            node2_edge.GetGradNode(),
            grad_node0));
  }
}

}  // namespace egr
