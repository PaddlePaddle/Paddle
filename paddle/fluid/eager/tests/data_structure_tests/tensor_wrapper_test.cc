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

#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/eager/tests/data_structure_tests/grad_node_test.h"
#include "paddle/fluid/eager/utils.h"

TEST(TensorWrapper, Basic) {
  VLOG(6) << "Test Full reserved";
  paddle::experimental::Tensor et1;
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, phi::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(paddle::platform::CPUPlace());
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 10.0f;
  et1.set_impl(dt);
  // Create grad node;
  auto grad_test_node0 = std::make_shared<eager_test::GradTestNode>(
      /* val */ 5.0, /* in_num */ 2, /* out_num */ 2);
  egr::Edge edge0(grad_test_node0, 1, 2);
  auto auto_grad0 = std::make_shared<egr::AutogradMeta>(edge0);
  et1.set_autograd_meta(auto_grad0);
  et1.set_name("et1");
  auto tw0 = egr::TensorWrapper(et1, true);
  auto recover_et1 = tw0.recover(std::make_shared<eager_test::GradTestNode>());
  CHECK_EQ(recover_et1.name(), std::string("et1"));
  CHECK_EQ(egr::EagerUtils::OutRankInfo(recover_et1).first,
           egr::EagerUtils::OutRankInfo(et1).first);
  CHECK_EQ(egr::EagerUtils::OutRankInfo(recover_et1).second,
           egr::EagerUtils::OutRankInfo(et1).second);
  VLOG(6) << "Test reconstruct";
  paddle::experimental::Tensor et2;
  phi::DenseTensorMeta meta2 =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, phi::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt2 = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta2);
  auto* dt_ptr2 = dt->mutable_data<float>(paddle::platform::CPUPlace());
  dt_ptr2[0] = 6.0f;
  dt_ptr2[1] = 11.0f;
  et2.set_impl(dt2);
  et2.set_name("et2");
  auto grad_test_node1 =
      std::make_shared<eager_test::GradTestNode>(/* val */ 5.0, 2, 2);
  egr::Edge edge1(grad_test_node1, 1, 2);
  auto auto_grad1 = std::make_shared<egr::AutogradMeta>(edge1);
  et2.set_autograd_meta(auto_grad1);
  auto tw1 = egr::TensorWrapper(et2, false);
  auto recover_et2 = tw1.recover(grad_test_node1);
  CHECK_EQ(recover_et2.name(), std::string("et2@Saved"));
  CHECK_EQ(egr::EagerUtils::OutRankInfo(recover_et2).first,
           egr::EagerUtils::OutRankInfo(et2).first);
  CHECK_EQ(egr::EagerUtils::OutRankInfo(recover_et2).second,
           egr::EagerUtils::OutRankInfo(et2).second);
  // Test Raw recover
  paddle::experimental::Tensor et3;
  auto tw2 = egr::TensorWrapper(et3, true);
  CHECK(
      tw2.recover(std::make_shared<eager_test::GradTestNode>()).initialized() ==
      false);
}
