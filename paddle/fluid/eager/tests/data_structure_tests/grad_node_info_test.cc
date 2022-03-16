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

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/tests/data_structure_tests/grad_node_test.h"
#include "paddle/phi/api/lib/utils/allocator.h"

TEST(GradNodeInfo, GradSlotMeta) {
  auto grad_slot = egr::GradSlotMeta();
  VLOG(6) << "Set SetStopGradient";
  grad_slot.SetStopGradient();
  CHECK(grad_slot.IsStopGradient() == true);
}

void TestGradNodeBase(bool is_remove_gradient_hook) {
  VLOG(6) << "Construct Grad Node";
  auto grad_test_node0 = std::make_shared<eager_test::GradTestNode>(
      /* val */ 5.0, /* in_num */ 2, /* out_num */ 2);
  auto grad_test_node1 = std::make_shared<eager_test::GradTestNode>();
  std::vector<std::vector<paddle::experimental::Tensor>> grads;
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, phi::make_ddim({1, 1}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(paddle::platform::CPUPlace());
  dt_ptr[0] = 5.0f;
  paddle::experimental::Tensor et1(dt);
  grads = {{et1}};
  VLOG(6) << "Test Grad Node Call";
  auto res = (*grad_test_node0)(grads);
  CHECK_EQ(std::dynamic_pointer_cast<phi::DenseTensor>(res[0][0].impl())
               ->data<float>()[0],
           6.0f);
  VLOG(6) << "Test Add Edges";
  egr::Edge tmp_edge0(grad_test_node1, 1, 2);
  auto auto_grad0 = std::make_shared<egr::AutogradMeta>(tmp_edge0);
  auto_grad0->SetStopGradient(false);

  egr::Edge tmp_edge1(grad_test_node1, 3, 4);
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>(tmp_edge1);
  et1.set_autograd_meta(auto_grad1);
  auto_grad1->SetStopGradient(false);
  grad_test_node0->AddEdges(auto_grad0.get(), 0);

  CHECK_EQ(grad_test_node0->GetEdges()[0][0].GetEdgeRankInfo().first,
           size_t(1));
  CHECK_EQ(grad_test_node0->GetEdges()[0][0].GetEdgeRankInfo().second,
           size_t(2));
  std::vector<egr::AutogradMeta*> metas = {auto_grad1.get()};

  grad_test_node0->AddEdges(&metas, 1);
  CHECK_EQ(grad_test_node0->GetEdges()[1][0].GetEdgeRankInfo().first,
           size_t(3));
  CHECK_EQ(grad_test_node0->GetEdges()[1][0].GetEdgeRankInfo().second,
           size_t(4));

  VLOG(6) << "Test Set Meta and Get Meta";
  auto_grad1->SetStopGradient(true);
  grad_test_node0->SetGradInMeta(et1, 0);
  grad_test_node0->SetGradInMeta({et1}, 1);
  grad_test_node0->SetGradOutMeta(et1, 0);
  grad_test_node0->SetGradOutMeta({et1}, 1);
  CHECK_EQ(grad_test_node0->InputMeta()[0].size(), size_t(1));
  CHECK_EQ(grad_test_node0->InputMeta()[1].size(), size_t(1));
  CHECK_EQ(grad_test_node0->InputMeta()[0][0].GetTensorMeta().dtype,
           meta.dtype);
  CHECK_EQ(grad_test_node0->InputMeta()[1][0].GetTensorMeta().dtype,
           meta.dtype);
  CHECK(grad_test_node0->OutputMeta()[0][0].IsStopGradient());
  CHECK(grad_test_node0->OutputMeta()[1][0].IsStopGradient());
  CHECK_EQ(grad_test_node0->OutputMeta()[0][0].GetTensorMeta().dtype,
           meta.dtype);
  CHECK_EQ(grad_test_node0->OutputMeta()[1][0].GetTensorMeta().dtype,
           meta.dtype);

  VLOG(6) << "Test Default Set Meta and Get Meta";
  auto grad_test_node2 = std::make_shared<eager_test::GradTestNode>(
      /* val */ 5.0, /* in_num */ 1, /* out_num */ 1);
  grad_test_node2->SetDefaultGradInOutMeta();
  CHECK_GT(grad_test_node2->OutputMeta()[0].size(), size_t(0));
  CHECK(grad_test_node2->OutputMeta()[0][0].IsStopGradient() == false);
  CHECK_EQ(grad_test_node2->OutputMeta()[0].size(), size_t(1));

  VLOG(6) << "Test Gradient Hook";
  auto gradient_hook = [](
      const paddle::experimental::Tensor& et) -> paddle::experimental::Tensor {
    paddle::experimental::Tensor res;
    phi::DenseTensorMeta meta =
        phi::DenseTensorMeta(phi::DataType::FLOAT32, phi::make_ddim({1, 1}));
    std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        meta);
    auto* dt_ptr = dt->mutable_data<float>(paddle::platform::CPUPlace());
    dt_ptr[0] = 6.0f;
    auto* et_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(et.impl())->data<float>();
    dt_ptr[0] += et_ptr[0];
    res.set_impl(dt);
    VLOG(6) << "Running Gradient Hook";
    return res;
  };
  int64_t hook_id = grad_test_node0->RegisterGradientHook(
      0, 0, std::make_shared<egr::CppTensorHook>(gradient_hook));

  if (is_remove_gradient_hook) {
    // Remove GradientHook
    grad_test_node0->RemoveGradientHook(hook_id);
  }

  // Check results
  auto grad_hook_res = grad_test_node0->ApplyGradientHooks(grads);
  CHECK_EQ(
      std::dynamic_pointer_cast<phi::DenseTensor>(grad_hook_res[0][0].impl())
          ->data<float>()[0],
      is_remove_gradient_hook ? 5.0 : 11.0);
}

TEST(GradNodeInfo, GradNodeBase) {
  TestGradNodeBase(true);
  TestGradNodeBase(false);
}

TEST(GradNodeInfo, Edge) {
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, phi::make_ddim({1, 1}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  paddle::experimental::Tensor et1(dt);

  auto grad_test_node0 = std::make_shared<eager_test::GradTestNode>(5, 2, 2);
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>();
  VLOG(6) << "Test Construct Edge";
  egr::Edge edge0 = egr::Edge();
  CHECK(edge0.IsInitialized() == false);
  egr::Edge edge1 = egr::Edge(grad_test_node0, size_t(0), size_t(0));
  CHECK(edge1.IsInitialized() == true);
  egr::Edge edge2 =
      egr::Edge(grad_test_node0, std::make_pair(size_t(1), size_t(0)));
  VLOG(6) << "Test Set Edge's Grad Node";
  auto* grad_node = edge1.GetGradNode();
  et1.set_autograd_meta(auto_grad1);
  grad_node->SetGradInMeta(et1, 0);

  CHECK_EQ(grad_node->InputMeta().size(), size_t(2));
  std::vector<egr::AutogradMeta*> metas = {auto_grad1.get()};
  CHECK(grad_node->InputMeta()[0][0].IsStopGradient() == true);
  VLOG(6) << "Test Get/Set Edge Rank Info";
  CHECK_EQ(edge2.GetEdgeRankInfo().first, size_t(1));
  CHECK_EQ(edge2.GetEdgeRankInfo().second, size_t(0));
  edge2.SetEdgeRankInfo(2, 3);
  CHECK_EQ(edge2.GetEdgeRankInfo().first, size_t(2));
  CHECK_EQ(edge2.GetEdgeRankInfo().second, size_t(3));
  edge2.SetEdgeRankInfo(std::make_pair(size_t(4), size_t(5)));
  CHECK_EQ(edge2.GetEdgeRankInfo().first, size_t(4));
  CHECK_EQ(edge2.GetEdgeRankInfo().second, size_t(5));
}
