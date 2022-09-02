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

#include "paddle/fluid/eager/eager_tensor.h"

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"

namespace eager_test {
using AbstractAutogradMeta = paddle::experimental::AbstractAutogradMeta;
class AutogradMetaTest : public AbstractAutogradMeta {
 public:
  explicit AutogradMetaTest(int val) : val_(val) {}
  int val_ = 0;
};
}  // namespace eager_test
TEST(Tensor, Constructor) {
  paddle::experimental::Tensor et1 = paddle::experimental::Tensor();
  paddle::experimental::Tensor et2 = paddle::experimental::Tensor("et2");

  CHECK_EQ(et1.defined(), false);
  CHECK_EQ(et2.name(), "et2");

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
  paddle::experimental::Tensor et3 = paddle::experimental::Tensor(dt);
  auto* et3_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et3.impl())->data<float>();
  CHECK_EQ(et3_ptr[0], 5.0f);
  CHECK_EQ(et3_ptr[1], 10.0f);
  // copy constructor
  paddle::experimental::Tensor et4(et3);
  auto* et4_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et4.impl())->data<float>();
  CHECK_EQ(et4_ptr[0], 5.0f);
  CHECK_EQ(et4_ptr[1], 10.0f);
  paddle::experimental::Tensor et5(std::move(et4));
  auto* et5_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et5.impl())->data<float>();
  CHECK_EQ(et5_ptr[0], 5.0f);
  CHECK_EQ(et5_ptr[1], 10.0f);
}

TEST(Tensor, MemberFunction) {
  paddle::experimental::Tensor et3;
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
  VLOG(6) << "Make Dense Tensor";
  et3.set_name("et3");
  VLOG(6) << "Set Name";
  CHECK_EQ(et3.name(), "et3");
  CHECK_EQ(et3.defined(), false);
  et3.set_impl(dt);
  VLOG(6) << "Set impl";
  CHECK_EQ(et3.initialized(), true);
  CHECK_EQ(et3.is_cpu(), true);
  CHECK_EQ(et3.is_gpu(), false);
  CHECK_EQ(et3.numel(), 2);
  auto expected_dim = phi::make_ddim({1, 2});
  CHECK_EQ(et3.dims(), expected_dim);
  CHECK_EQ(et3.type(), paddle::experimental::DataType::FLOAT32);
  CHECK_EQ(et3.layout(), paddle::experimental::DataLayout::NCHW);
  CHECK(paddle::platform::is_cpu_place(et3.place()));
  VLOG(6) << "Get impl";
  auto* dt3_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et3.impl())->data<float>();
  CHECK_EQ(dt3_ptr[0], 5.0f);
  CHECK_EQ(dt3_ptr[1], 10.0f);
  paddle::experimental::Tensor et4 = et3;
  VLOG(6) << "copy =";
  CHECK(et4.initialized() == true);
  auto* dt4_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et4.impl())->data<float>();
  CHECK_EQ(dt4_ptr[0], 5.0f);
  CHECK_EQ(dt4_ptr[1], 10.0f);
  VLOG(6) << "move =";
  paddle::experimental::Tensor et5 = std::move(et4);
  auto* dt5_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et5.impl())->data<float>();
  CHECK_EQ(dt5_ptr[0], 5.0f);
  CHECK_EQ(dt5_ptr[1], 10.0f);
  VLOG(6) << "AutogradMeta";
  auto autograd_meta_test = std::make_shared<eager_test::AutogradMetaTest>(2);
  et3.set_autograd_meta(autograd_meta_test);
  auto* tmp_autograd_meta_test =
      static_cast<eager_test::AutogradMetaTest*>(et3.get_autograd_meta());
  CHECK_EQ(tmp_autograd_meta_test->val_, 2);
}

TEST(EagerVariable, Constructor) {
  paddle::experimental::Tensor t3;
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
  VLOG(6) << "Make Dense Tensor";
  t3.set_name("t3");
  VLOG(6) << "Set Name";
  CHECK_EQ(t3.name(), "t3");
  CHECK_EQ(t3.defined(), false);
  t3.set_impl(dt);

  egr::EagerVariable et3 = egr::EagerVariable(t3);
  VLOG(6) << "SyncToVar";
  CHECK_EQ(et3.Var().Get<paddle::framework::LoDTensor>().data<float>()[0],
           5.0f);
  CHECK_EQ(et3.Var().Get<paddle::framework::LoDTensor>().data<float>()[1],
           10.0f);
  VLOG(6) << "SyncToTensor";
  paddle::experimental::Tensor t4;
  t4.set_impl(et3.GetTensorBase());
  CHECK(t4.initialized() == true);
  VLOG(6) << "Check Tensor";
  auto* dt3_tmp_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(t4.impl())->data<float>();
  CHECK_EQ(dt3_tmp_ptr[0], 5.0f);
  CHECK_EQ(dt3_tmp_ptr[1], 10.0f);
  t4.reset();
  CHECK(t4.defined() == false);

  VLOG(6) << "Check Tensor Copy_";
  std::vector<int64_t> rows = {1, 2};
  std::vector<int64_t> dims = {2};
  paddle::experimental::Tensor t7(std::make_shared<phi::SelectedRows>(rows, 2));
  std::dynamic_pointer_cast<phi::SelectedRows>(t7.impl())
      ->mutable_value()
      ->Resize(phi::make_ddim(dims));
  auto* dt7_tmp_ptr = std::dynamic_pointer_cast<phi::SelectedRows>(t7.impl())
                          ->mutable_value()
                          ->mutable_data<float>(paddle::platform::CPUPlace());
  dt7_tmp_ptr[0] = 6.0f;
  dt7_tmp_ptr[1] = 11.0f;

  paddle::experimental::Tensor t8;
  paddle::experimental::Tensor t5;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  paddle::experimental::Tensor t6;
  paddle::experimental::Tensor t9;
  VLOG(6) << "Check Tensor Copy_ Selected Rows";
  t8.copy_(t7, paddle::platform::CUDAPlace(0), false);
  t9.copy_(t8, paddle::platform::CPUPlace(), false);
  auto* dt9_tmp_ptr = std::dynamic_pointer_cast<phi::SelectedRows>(t9.impl())
                          ->value()
                          .data<float>();
  CHECK_EQ(dt9_tmp_ptr[0], 6.0f);
  CHECK_EQ(dt9_tmp_ptr[1], 11.0f);
  CHECK_EQ(std::dynamic_pointer_cast<phi::SelectedRows>(t9.impl())->height(),
           2);

  VLOG(6) << "Check Tensor Copy_ Dense Tensor";
  t5.copy_(t3, paddle::platform::CUDAPlace(0), false);
  t6.copy_(t5, paddle::platform::CPUPlace(), false);
  auto* dt6_tmp_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(t6.impl())->data<float>();
  CHECK_EQ(dt6_tmp_ptr[0], 5.0f);
  CHECK_EQ(dt6_tmp_ptr[1], 10.0f);
#else
  t5.copy_(t3, paddle::platform::CPUPlace(), false);
  auto* dt5_tmp_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(t5.impl())->data<float>();
  CHECK_EQ(dt5_tmp_ptr[0], 5.0f);
  CHECK_EQ(dt5_tmp_ptr[1], 10.0f);
#endif

  VLOG(6) << "Finish";
}

TEST(EagerVariable, DataLayout) {
  paddle::experimental::Tensor tensor;
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({1, 1, 1, 1}),
                           paddle::experimental::DataLayout::UNDEFINED);
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(paddle::platform::CPUPlace());
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 5.0f;
  dt_ptr[2] = 5.0f;
  dt_ptr[3] = 5.0f;
  tensor.set_impl(dt);
  auto eager_var = std::make_shared<egr::EagerVariable>(tensor);
  auto layout = paddle::imperative::GetDataLayout(eager_var);
  CHECK_EQ(layout, paddle::experimental::DataLayout::UNDEFINED);
  paddle::imperative::SetDataLayout(eager_var,
                                    paddle::experimental::DataLayout::NCHW);
  layout = paddle::imperative::GetDataLayout(eager_var);
  CHECK_EQ(layout, paddle::experimental::DataLayout::NCHW);
}

TEST(VariableCompatTensor, MemberFunction) {
  egr::VariableCompatTensor var_tensor;
  // test GetMutable and Get
  var_tensor.GetMutable<paddle::framework::Vocab>();
  auto& vocab = var_tensor.Get<paddle::framework::Vocab>();
  EXPECT_EQ(vocab.size(), 0UL);
  bool caught_exception = false;
  try {
    var_tensor.GetMutable<paddle::framework::Strings>();
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("The Variable type must be") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  // test Type and IsType
  EXPECT_TRUE(var_tensor.IsType<paddle::framework::Vocab>());
  EXPECT_EQ(var_tensor.Type(),
            static_cast<int>(paddle::framework::proto::VarType::VOCAB));
  // test valid and initialized
  EXPECT_TRUE(var_tensor.IsInitialized());
  EXPECT_TRUE(var_tensor.valid());
  EXPECT_TRUE(var_tensor.initialized());
  // test name
  EXPECT_EQ(var_tensor.name(), "VariableCompatTensor");
  // test other throw error methods
  caught_exception = false;
  try {
    var_tensor.numel();
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("numel") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  caught_exception = false;
  try {
    var_tensor.dims();
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("dims") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  caught_exception = false;
  try {
    var_tensor.dtype();
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("dtype") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  caught_exception = false;
  try {
    var_tensor.layout();
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("layout") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  caught_exception = false;
  try {
    var_tensor.place();
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("place") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  caught_exception = false;
  try {
    var_tensor.AllocateFrom(nullptr, phi::DataType::UNDEFINED);
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("AllocateFrom") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  // test Clear
  var_tensor.Clear();
  EXPECT_FALSE(var_tensor.IsInitialized());
}
