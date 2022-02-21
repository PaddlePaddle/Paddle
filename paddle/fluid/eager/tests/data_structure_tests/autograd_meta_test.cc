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
#include "paddle/phi/api/lib/utils/allocator.h"

TEST(AutogradMeta, Constructor) {
  paddle::experimental::Tensor et1;
  auto auto_grad = std::make_shared<egr::AutogradMeta>();
  et1.set_autograd_meta(auto_grad);
  auto* tmp_auto = static_cast<egr::AutogradMeta*>(et1.get_autograd_meta());
  CHECK_EQ(tmp_auto->OutRankInfo().first, size_t(0));
  CHECK_EQ(tmp_auto->OutRankInfo().second, size_t(0));
  CHECK(tmp_auto->IsInitialized() == false);
}

TEST(AutogradMeta, MemberFunction) {
  paddle::experimental::Tensor et1;
  auto auto_grad = std::make_shared<egr::AutogradMeta>();
  et1.set_autograd_meta(auto_grad);
  auto* tmp_auto = static_cast<egr::AutogradMeta*>(et1.get_autograd_meta());
  VLOG(6) << "Test Grad";
  CHECK(tmp_auto->Grad().defined() == false);
  auto* grad_t = tmp_auto->MutableGrad();
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
  grad_t->set_impl(dt);
  VLOG(6) << "Test Mutable Grad";
  auto impl_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(tmp_auto->Grad().impl());
  CHECK_EQ(impl_ptr->data<float>()[0], 5.0f);
  CHECK_EQ(impl_ptr->data<float>()[1], 10.0f);
  VLOG(6) << "Test IsInitialized";
  CHECK(tmp_auto->IsInitialized() == false);
  VLOG(6) << "Test GradNodeSetter Getter";
  auto grad_node = std::make_shared<eager_test::GradTestNode>();
  tmp_auto->SetGradNode(grad_node);
  CHECK(tmp_auto->IsInitialized() == true);
  auto tmp_grad_node = tmp_auto->GetMutableGradNode();
  std::dynamic_pointer_cast<eager_test::GradTestNode>(tmp_grad_node)->val_ =
      5.0;
  CHECK_EQ(dynamic_cast<eager_test::GradTestNode*>(tmp_auto->GradNode())->val_,
           5.0);
  VLOG(6) << "Test rank Setter Getter";
  CHECK_EQ(tmp_auto->OutRankInfo().first, size_t(0));
  CHECK_EQ(tmp_auto->OutRankInfo().second, size_t(0));
  tmp_auto->SetSingleOutRankWithSlot(2, 3);
  CHECK_EQ(tmp_auto->OutRankInfo().first, size_t(2));
  CHECK_EQ(tmp_auto->OutRankInfo().second, size_t(3));
  VLOG(6) << "Test stop gradient Setter Getter";
  CHECK_EQ(tmp_auto->NumericStopGradient(), -1);
  tmp_auto->SetStopGradient(true);
  CHECK(tmp_auto->StopGradient() == true);
  VLOG(6) << "Test Persistable Setter Getter";
  CHECK(tmp_auto->Persistable() == false);
  tmp_auto->SetPersistable(true);
  CHECK(tmp_auto->Persistable() == true);
}
