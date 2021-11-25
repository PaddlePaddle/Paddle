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

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/pten/api/lib/utils/allocator.h"

namespace eager_test {
using AbstractAutogradMeta = paddle::experimental::AbstractAutogradMeta;
class AutogradMetaTest : public AbstractAutogradMeta {
 public:
  explicit AutogradMetaTest(int val) : val_(val) {}
  int val_ = 0;
};
}
TEST(EagerTensor, Constructor) {
  egr::EagerTensor et1 = egr::EagerTensor();
  egr::EagerTensor et2 = egr::EagerTensor("et2");

  CHECK_EQ(et1.defined(), false);
  CHECK_EQ(et2.name(), "et2");

  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({1, 2}));
  std::shared_ptr<pten::DenseTensor> dt = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  auto* dt_ptr = dt->mutable_data<float>();
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 10.0f;
  egr::EagerTensor et3 = egr::EagerTensor(dt);
  auto* et3_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(et3.impl())->data<float>();
  CHECK_EQ(et3_ptr[0], 5.0f);
  CHECK_EQ(et3_ptr[1], 10.0f);
  // copy constructor
  egr::EagerTensor et4(et3);
  auto* et4_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(et4.impl())->data<float>();
  CHECK_EQ(et4_ptr[0], 5.0f);
  CHECK_EQ(et4_ptr[1], 10.0f);
  egr::EagerTensor et5(std::move(et4));
  auto* et5_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(et5.impl())->data<float>();
  CHECK_EQ(et5_ptr[0], 5.0f);
  CHECK_EQ(et5_ptr[1], 10.0f);
}

TEST(EagerTensor, MemberFunction) {
  egr::EagerTensor et3;
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({1, 2}));
  std::shared_ptr<pten::DenseTensor> dt = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  auto* dt_ptr = dt->mutable_data<float>();
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 10.0f;
  VLOG(6) << "Make Dense Tensor";
  et3.set_name("et3");
  VLOG(6) << "Set Name";
  CHECK_EQ(et3.name(), "et3");
  CHECK_EQ(et3.defined(), false);
  et3.set_impl(dt);
  VLOG(6) << "Set impl";
  CHECK_EQ(et3.initialized(), true);
  CHECK_EQ(et3.is_cpu(), true);
  CHECK_EQ(et3.is_cuda(), false);
  CHECK_EQ(et3.numel(), 2);
  auto expected_dim = paddle::framework::make_ddim({1, 2});
  CHECK_EQ(et3.shape(), expected_dim);
  CHECK_EQ(et3.type(), paddle::experimental::DataType::FLOAT32);
  CHECK_EQ(et3.layout(), paddle::experimental::DataLayout::NCHW);
  CHECK(paddle::platform::is_cpu_place(et3.place()));
  VLOG(6) << "Get impl";
  auto* dt3_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(et3.impl())->data<float>();
  CHECK_EQ(dt3_ptr[0], 5.0f);
  CHECK_EQ(dt3_ptr[1], 10.0f);
  egr::EagerTensor et4 = et3;
  VLOG(6) << "copy =";
  CHECK(et4.initialized() == true);
  auto* dt4_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(et4.impl())->data<float>();
  CHECK_EQ(dt4_ptr[0], 5.0f);
  CHECK_EQ(dt4_ptr[1], 10.0f);
  VLOG(6) << "move =";
  egr::EagerTensor et5 = std::move(et4);
  auto* dt5_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(et5.impl())->data<float>();
  CHECK_EQ(dt5_ptr[0], 5.0f);
  CHECK_EQ(dt5_ptr[1], 10.0f);
  VLOG(6) << "AutogradMeta";
  auto autograd_meta_test = std::make_shared<eager_test::AutogradMetaTest>(2);
  et3.set_autograd_meta(autograd_meta_test);
  auto* tmp_autograd_meta_test =
      static_cast<eager_test::AutogradMetaTest*>(et3.get_autograd_meta());
  CHECK_EQ(tmp_autograd_meta_test->val_, 2);
  VLOG(6) << "SyncToVar";
  et3.SyncToVar();
  CHECK_EQ(et3.Var().Get<paddle::framework::LoDTensor>().data<float>()[0],
           5.0f);
  CHECK_EQ(et3.Var().Get<paddle::framework::LoDTensor>().data<float>()[1],
           10.0f);
  VLOG(6) << "SyncToTensor";
  CHECK(et3.initialized() == false);
  et3.SyncToTensor();
  CHECK(et3.initialized() == true);
  VLOG(6) << "Check Tensor";
  auto* dt3_tmp_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(et3.impl())->data<float>();
  CHECK_EQ(dt3_tmp_ptr[0], 5.0f);
  CHECK_EQ(dt3_tmp_ptr[1], 10.0f);
  et3.reset();
  CHECK(et3.defined() == false);
  VLOG(6) << "Finish";
}
