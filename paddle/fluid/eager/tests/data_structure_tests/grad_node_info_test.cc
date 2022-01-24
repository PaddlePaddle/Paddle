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
#include "paddle/fluid/eager/tests/data_structure_tests/grad_node_test.h"
#include "paddle/pten/api/lib/utils/allocator.h"

TEST(GradNodeInfo, GradSlotMeta) {
  auto grad_slot = egr::GradSlotMeta();
  CHECK(grad_slot.IsInitialized() == false);
  VLOG(6) << "Init GradSlotMeta";
  grad_slot.Init(2);
  CHECK(grad_slot.IsInitialized() == true);
  VLOG(6) << "Set SetStopGradient";
  grad_slot.SetStopGradient(0);
  CHECK(grad_slot.IsStopGradient(0) == true);
  CHECK_EQ(grad_slot.Size(), 2);
}

TEST(GradNodeInfo, GradNodeBase) {
  VLOG(6) << "Construct Grad Node";
  auto grad_test_node0 = std::make_shared<eager_test::GradTestNode>(
      /* val */ 5.0, /* in_num */ 2, /* out_num */ 2);
  auto grad_test_node1 = std::make_shared<eager_test::GradTestNode>();
  std::vector<std::vector<egr::EagerTensor>> grads;
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({1, 1}));
  std::shared_ptr<pten::DenseTensor> dt = std::make_shared<pten::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>();
  dt_ptr[0] = 5.0f;
  egr::EagerTensor et1(dt);
  grads = {{et1}};
  VLOG(6) << "Test Grad Node Call";
  auto res = (*grad_test_node0)(grads);
  CHECK_EQ(std::dynamic_pointer_cast<pten::DenseTensor>(res[0][0].impl())
               ->data<float>()[0],
           6.0f);
  VLOG(6) << "Test Add Edges";
  egr::Edge edge0(grad_test_node1, 1, 2);
  auto auto_grad0 = std::make_shared<egr::AutogradMeta>(edge0);
  auto_grad0->SetStopGradient(false);
  egr::Edge edge1(grad_test_node1, 3, 4);
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>(edge1);
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
  grad_test_node0->SetGradInMeta(metas, 0);
  grad_test_node0->SetGradInMeta(auto_grad1.get(), 1);
  grad_test_node0->SetGradOutMeta(metas, 0);
  grad_test_node0->SetGradOutMeta(auto_grad1.get(), 1);
  CHECK_EQ(grad_test_node0->InputMeta()[0].Size(), 1);
  CHECK_EQ(grad_test_node0->InputMeta()[1].Size(), 1);
  CHECK(grad_test_node0->OutputMeta()[0].IsStopGradient(0));
  CHECK(grad_test_node0->OutputMeta()[1].IsStopGradient(0));

  VLOG(6) << "Test Default Set Meta and Get Meta";
  auto grad_test_node2 = std::make_shared<eager_test::GradTestNode>(
      /* val */ 5.0, /* in_num */ 1, /* out_num */ 1);
  grad_test_node2->SetDefaultGradInOutMeta();
  CHECK(grad_test_node2->OutputMeta()[0].IsInitialized());
  CHECK(grad_test_node2->OutputMeta()[0].IsStopGradient(0) == false);
  CHECK_EQ(grad_test_node2->OutputMeta()[0].Size(), 1);

  VLOG(6) << "Test Gradient Hook";
  auto gradient_hook = [](const egr::EagerTensor& et) -> egr::EagerTensor {
    egr::EagerTensor res;
    pten::DenseTensorMeta meta = pten::DenseTensorMeta(
        pten::DataType::FLOAT32, paddle::framework::make_ddim({1, 1}));
    std::shared_ptr<pten::DenseTensor> dt = std::make_shared<pten::DenseTensor>(
        std::make_unique<paddle::experimental::DefaultAllocator>(
            paddle::platform::CPUPlace())
            .get(),
        meta);
    auto* dt_ptr = dt->mutable_data<float>();
    dt_ptr[0] = 6.0f;
    auto* et_ptr =
        std::dynamic_pointer_cast<pten::DenseTensor>(et.impl())->data<float>();
    dt_ptr[0] += et_ptr[0];
    res.set_impl(dt);
    VLOG(6) << "Running Gradient Hook";
    return res;
  };
  grad_test_node0->RegisterGradientHook(0, 0, gradient_hook);
  // 5 + 6
  auto grad_hook_res = grad_test_node0->ApplyGradientHooks(grads);
  CHECK_EQ(
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_hook_res[0][0].impl())
          ->data<float>()[0],
      11.0);

  VLOG(6) << "Test Reduce Hook";
  auto reduce_hook = [&](void) -> void {
    auto* et_ptr = std::dynamic_pointer_cast<pten::DenseTensor>(et1.impl())
                       ->mutable_data<float>();
    et_ptr[0] = 100.0;
    VLOG(6) << "Running Reduce Hook";
  };
  grad_test_node0->RegisterReduceHook(reduce_hook);
  grad_test_node0->ApplyReduceHooks();
  CHECK_EQ(std::dynamic_pointer_cast<pten::DenseTensor>(et1.impl())
               ->data<float>()[0],
           100.0);
}

TEST(GradNodeInfo, Edge) {
  auto grad_test_node0 = std::make_shared<eager_test::GradTestNode>(5, 2, 2);
  VLOG(6) << "Test Construct Edge";
  egr::Edge edge0 = egr::Edge();
  CHECK(edge0.IsInitialized() == false);
  egr::Edge edge1 = egr::Edge(grad_test_node0, size_t(0), size_t(0));
  CHECK(edge1.IsInitialized() == true);
  egr::Edge edge2 =
      egr::Edge(grad_test_node0, std::make_pair(size_t(1), size_t(0)));
  VLOG(6) << "Test Set Edge's Grad Node";
  auto* grad_node = edge1.GetGradNode();
  CHECK_EQ(grad_node->InputMeta().size(), size_t(2));
  auto mt_grad_node = edge1.GetMutableGradNode();
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>();
  std::vector<egr::AutogradMeta*> metas = {auto_grad1.get()};
  // Uninitialized AutogradMeta indicates
  mt_grad_node->SetGradInMeta(metas, 0);
  CHECK(grad_node->InputMeta()[0].IsStopGradient(0) == true);
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
