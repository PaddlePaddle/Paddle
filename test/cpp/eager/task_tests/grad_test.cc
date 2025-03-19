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
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "test/cpp/eager/test_utils.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);

namespace egr {

TEST(Grad, SingleNodeEmptyGrad) {
  // Prepare Device Contexts
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor (output)
  paddle::Tensor output_tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);

  // Create input tensor
  const paddle::Tensor leaf_tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        true /*is_leaf*/);

  {
    // Create Scale Node
    auto node0_ptr = std::make_shared<GradNodeScale>(1, 1);
    node0_ptr->SetAttributes_scale(5.0 /*scale*/);

    // Set grad in/out meta
    node0_ptr->SetDefaultGradInOutMeta();

    // Output_tensor set GradNode、OutRank、StopGradient properties
    AutogradMeta* auto_grad_meta = EagerUtils::autograd_meta(&output_tensor);
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta->SetStopGradient(false);

    // Get autograd_meta from input tensor
    AutogradMeta* auto_grad_meta1 =
        EagerUtils::unsafe_autograd_meta(leaf_tensor);

    // Connect Tensor and AccumulationNode via AutoGradMeta
    auto acc_node_ptr =
        std::make_shared<egr::GradNodeAccumulation>(auto_grad_meta1);

    // input tensor set GradNode、OutRank、StopGradient properties
    auto_grad_meta1->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta1->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta1->SetStopGradient(false);

    // grad_node Add Edges
    std::vector<egr::AutogradMeta*> res = {auto_grad_meta1};
    node0_ptr->SetGradOutMeta(leaf_tensor, 0);
  }
  std::vector<paddle::Tensor> outs = {output_tensor};

  // Run Grad
  auto result = Grad(outs, {leaf_tensor}, {});
  // Check Output Value
  eager_test::CompareTensorWithValue<float>(result[0], 5.0);
}

TEST(Grad, SingleNodeCustomGrad) {
  // Prepare Device Contexts
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::Tensor> target_tensors;
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));

  std::vector<paddle::Tensor> grad_tensors;
  // Create Grad Tensor
  paddle::Tensor grad_tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        10.0 /*value*/,
                                        false /*is_leaf*/);
  grad_tensors.emplace_back(std::move(grad_tensor));

  paddle::Tensor leaf_tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        true /*is_leaf*/);

  {
    // Create Scale Node
    auto node0_ptr = std::make_shared<GradNodeScale>(1, 1);
    node0_ptr->SetAttributes_scale(5.0 /*scale*/);

    // Set grad in/out meta
    node0_ptr->SetDefaultGradInOutMeta();

    // Connect Tensor and Node via AutoGradMeta
    AutogradMeta* auto_grad_meta =
        EagerUtils::autograd_meta(&(target_tensors[0]));
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta->SetStopGradient(false);

    AutogradMeta* auto_grad_meta1 = EagerUtils::autograd_meta(&leaf_tensor);
    // Connect Tensor and AccumulationNode via AutoGradMeta
    auto acc_node_ptr =
        std::make_shared<egr::GradNodeAccumulation>(auto_grad_meta1);

    auto_grad_meta1->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta1->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta1->SetStopGradient(false);
    std::vector<egr::AutogradMeta*> res = {auto_grad_meta1};
    node0_ptr->SetGradOutMeta(leaf_tensor, 0);
  }

  auto result = Grad(target_tensors, {leaf_tensor}, grad_tensors);

  // Check Output Value
  eager_test::CompareTensorWithValue<float>(result[0], 50.0);
}

/*
Node1
  |
Node0
  |
 { } // empty grad tensor
*/
TEST(Grad, LinearNodes) {
  // Prepare Device Contexts
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Target Tensor
  std::vector<paddle::Tensor> target_tensors;
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));

  paddle::Tensor leaf_tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        true /*is_leaf*/);
  {
    // Create Node0
    auto node0_ptr = std::make_shared<GradNodeScale>(1, 1);
    node0_ptr->SetAttributes_scale(5.0 /*scale*/);

    // Set grad in/out meta for node0
    node0_ptr->SetDefaultGradInOutMeta();

    // Create Node1
    auto node1_ptr = std::make_shared<GradNodeScale>(1, 1);
    node1_ptr->SetAttributes_scale(10.0 /*scale*/);

    // Set grad in/out meta for node1
    node1_ptr->SetDefaultGradInOutMeta();

    // Connect Input Tensor and Node0 via AutoGradMeta
    AutogradMeta* auto_grad_meta =
        EagerUtils::autograd_meta(&(target_tensors[0]));
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta->SetStopGradient(false);
    // Connect Node0 -> Node1 via Edge
    auto tmp_tensor = paddle::Tensor();
    auto* meta0 = EagerUtils::autograd_meta(&tmp_tensor);
    meta0->SetStopGradient(false);
    meta0->SetSingleOutRankWithSlot(0, 0);
    meta0->SetGradNode(node1_ptr);
    node0_ptr->SetGradOutMeta(tmp_tensor, 0);

    AutogradMeta* auto_grad_meta1 = EagerUtils::autograd_meta(&leaf_tensor);
    // Connect Tensor and AccumulationNode via AutoGradMeta
    auto acc_node_ptr =
        std::make_shared<egr::GradNodeAccumulation>(auto_grad_meta1);

    auto_grad_meta1->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta1->SetSingleOutRankWithSlot(0, 0);

    auto_grad_meta1->SetStopGradient(false);
    node1_ptr->SetGradOutMeta(leaf_tensor, 0);
  }

  // Use Empty Grad Tensor
  auto result = Grad(target_tensors, {leaf_tensor}, {});

  // Check Output Value
  eager_test::CompareTensorWithValue<float>(result[0], 50.0);
}

/*
    Node2
    |   |
Node0   Node1
  |      |
 in0   in1
*/
TEST(Grad, WithAccumulation) {
  // Prepare Device Contexts
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  std::vector<paddle::Tensor> target_tensors;
  paddle::Tensor tensor0 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);
  paddle::Tensor tensor1 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor0));
  target_tensors.emplace_back(std::move(tensor1));

  // Create Grad Tensor
  std::vector<paddle::Tensor> grad_tensors;
  paddle::Tensor grad_tensor0 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        false /*is_leaf*/);
  paddle::Tensor grad_tensor1 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        10.0 /*value*/,
                                        false /*is_leaf*/);
  grad_tensors.emplace_back(std::move(grad_tensor0));
  grad_tensors.emplace_back(std::move(grad_tensor1));

  paddle::Tensor leaf_tensor;
  {
    // Create Node0
    auto node0_ptr = std::make_shared<GradNodeScale>(1, 1);
    node0_ptr->SetAttributes_scale(5.0 /*scale*/);
    node0_ptr->SetDefaultGradInOutMeta();

    // Create Node1
    auto node1_ptr = std::make_shared<GradNodeScale>(1, 1);
    node1_ptr->SetAttributes_scale(10.0 /*scale*/);
    node1_ptr->SetDefaultGradInOutMeta();
    // Create Node2
    auto node2_ptr = std::make_shared<GradNodeScale>(1, 1);
    node2_ptr->SetAttributes_scale(20.0 /*scale*/);
    node2_ptr->SetDefaultGradInOutMeta();
    // Connect Inp0 and Node0 via AutoGradMeta
    AutogradMeta* auto_grad_meta0 =
        EagerUtils::autograd_meta(&(target_tensors[0]));
    auto_grad_meta0->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(node0_ptr));
    auto_grad_meta0->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta0->SetStopGradient(false);
    // Connect Inp1 and Node1 via AutoGradMeta
    AutogradMeta* auto_grad_meta1 =
        EagerUtils::autograd_meta(&(target_tensors[1]));
    auto_grad_meta1->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(node1_ptr));
    auto_grad_meta1->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta1->SetStopGradient(false);

    // Connect Node0 -> Node2 via Edge
    auto tmp_tensor0 = paddle::Tensor();
    auto* meta0 = EagerUtils::autograd_meta(&tmp_tensor0);
    meta0->SetStopGradient(false);
    meta0->SetSingleOutRankWithSlot(0, 0);
    meta0->SetGradNode(node2_ptr);
    node0_ptr->SetGradOutMeta(tmp_tensor0, 0);

    // Connect Node1 -> Node2 via Edge
    auto tmp_tensor1 = paddle::Tensor();
    auto meta1 = EagerUtils::autograd_meta(&tmp_tensor1);
    meta1->SetStopGradient(false);
    meta1->SetSingleOutRankWithSlot(0, 0);
    meta1->SetGradNode(node2_ptr);
    node1_ptr->SetGradOutMeta(tmp_tensor1, 0);

    AutogradMeta* auto_grad_meta2 = EagerUtils::autograd_meta(&leaf_tensor);
    // Connect Tensor and AccumulationNode via AutoGradMeta
    auto acc_node_ptr =
        std::make_shared<egr::GradNodeAccumulation>(auto_grad_meta2);

    auto_grad_meta2->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta2->SetSingleOutRankWithSlot(0, 0);

    auto_grad_meta2->SetStopGradient(false);
    node2_ptr->SetGradOutMeta(leaf_tensor, 0);
  }

  auto result = Grad(target_tensors, {leaf_tensor}, grad_tensors);

  eager_test::CompareTensorWithValue<float>(result[0], 2500.0);
}

}  // namespace egr
