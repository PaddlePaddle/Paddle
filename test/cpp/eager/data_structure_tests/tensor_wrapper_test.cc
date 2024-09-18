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

#include "paddle/fluid/eager/tensor_wrapper.h"

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/eager/utils.h"
#include "test/cpp/eager/data_structure_tests/grad_node_test.h"

TEST(TensorWrapper, Basic) {
  VLOG(6) << "Test Full reserved";
  paddle::Tensor et1;
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, common::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(phi::CPUPlace());
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
  auto tw0 = egr::TensorWrapper(et1);
  auto recover_et1 = tw0.recover();
  if (VLOG_IS_ON(7)) {
    PADDLE_ENFORCE_EQ(
        recover_et1.name(),
        std::string("et1@saved"),
        common::errors::InvalidArgument(
            "Recovered tensor name should be 'et1@saved', but received %s.",
            recover_et1.name().c_str()));
  }
  PADDLE_ENFORCE_EQ(egr::EagerUtils::OutRankInfo(recover_et1).first,
                    egr::EagerUtils::OutRankInfo(et1).first,
                    common::errors::InvalidArgument(
                        "The OutRankInfo first element of the recovered tensor "
                        "does not match the original tensor."));
  PADDLE_ENFORCE_EQ(egr::EagerUtils::OutRankInfo(recover_et1).second,
                    egr::EagerUtils::OutRankInfo(et1).second,
                    common::errors::InvalidArgument(
                        "The OutRankInfo second element of the recovered "
                        "tensor does not match the original tensor."));
  VLOG(6) << "Test reconstruct";
  paddle::Tensor et2;
  phi::DenseTensorMeta meta2 =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, common::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt2 = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta2);
  auto* dt_ptr2 = dt->mutable_data<float>(phi::CPUPlace());
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
  auto recover_et2 = tw1.recover();
  if (VLOG_IS_ON(7)) {
    PADDLE_ENFORCE_EQ(
        recover_et2.name(),
        std::string("et2@Saved"),
        common::errors::InvalidArgument(
            "Recovered tensor name should be 'et2@Saved', but received %s.",
            recover_et2.name().c_str()));
  }
  PADDLE_ENFORCE_EQ(egr::EagerUtils::OutRankInfo(recover_et2).first,
                    egr::EagerUtils::OutRankInfo(et2).first,
                    common::errors::InvalidArgument(
                        "The OutRankInfo first element of the recovered tensor "
                        "does not match the original tensor."));
  PADDLE_ENFORCE_EQ(egr::EagerUtils::OutRankInfo(recover_et2).second,
                    egr::EagerUtils::OutRankInfo(et2).second,
                    common::errors::InvalidArgument(
                        "The OutRankInfo second element of the recovered "
                        "tensor does not match the original tensor."));
  // Test Raw recover
  paddle::Tensor et3;
  auto tw2 = egr::TensorWrapper(et3);
  PADDLE_ENFORCE_EQ(
      tw2.recover().initialized(),
      false,
      common::errors::Fatal(
          "Variable `tw2` should not be initialized after recover"));
}
