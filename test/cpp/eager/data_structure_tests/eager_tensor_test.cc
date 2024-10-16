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
#include "paddle/common/layout.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace eager_test {
using AbstractAutogradMeta = paddle::AbstractAutogradMeta;
class AutogradMetaTest : public AbstractAutogradMeta {
 public:
  explicit AutogradMetaTest(int val) : val_(val) {}
  int val_ = 0;
};
}  // namespace eager_test

TEST(Tensor, Constructor) {
  paddle::Tensor et1 = paddle::Tensor();
  paddle::Tensor et2 = paddle::Tensor("et2");

  PADDLE_ENFORCE_EQ(et1.defined(),
                    false,
                    common::errors::InvalidArgument("Tensor et1 should be "
                                                    "undefined."));
  PADDLE_ENFORCE_EQ(et2.name(),
                    "et2",
                    common::errors::InvalidArgument("Tensor name should be "
                                                    "'et2'."));

  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, common::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(phi::CPUPlace());
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 10.0f;
  paddle::Tensor et3 = paddle::Tensor(dt);
  auto* et3_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et3.impl())->data<float>();
  PADDLE_ENFORCE_EQ(et3_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(et3_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
  // copy constructor
  paddle::Tensor et4(et3);
  auto* et4_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et4.impl())->data<float>();
  PADDLE_ENFORCE_EQ(et4_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(et4_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
  paddle::Tensor et5(std::move(et4));
  auto* et5_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et5.impl())->data<float>();
  PADDLE_ENFORCE_EQ(et5_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(et5_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
}

TEST(Tensor, MemberFunction) {
  paddle::Tensor et3;
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, common::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(phi::CPUPlace());
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 10.0f;
  VLOG(6) << "Make Dense Tensor";
  et3.set_name("et3");
  VLOG(6) << "Set Name";
  PADDLE_ENFORCE_EQ(et3.name(),
                    "et3",
                    common::errors::InvalidArgument("Tensor name should be "
                                                    "'et3'."));
  PADDLE_ENFORCE_EQ(et3.defined(),
                    false,
                    common::errors::InvalidArgument("Tensor et3 should be "
                                                    "undefined."));
  et3.set_impl(dt);
  VLOG(6) << "Set impl";
  PADDLE_ENFORCE_EQ(et3.initialized(),
                    true,
                    common::errors::InvalidArgument("Tensor et3 should be "
                                                    "initialized."));
  PADDLE_ENFORCE_EQ(et3.is_cpu(),
                    true,
                    common::errors::InvalidArgument("Tensor et3 should be "
                                                    "on CPU."));
  PADDLE_ENFORCE_EQ(et3.is_gpu(),
                    false,
                    common::errors::InvalidArgument("Tensor et3 should not be "
                                                    "on GPU."));
  PADDLE_ENFORCE_EQ(et3.numel(),
                    2,
                    common::errors::InvalidArgument("Tensor et3 should have "
                                                    "2 elements."));
  auto expected_dim = common::make_ddim({1, 2});
  PADDLE_ENFORCE_EQ(
      et3.dims(),
      expected_dim,
      common::errors::InvalidArgument("Tensor dimensions should be "
                                      "{1, 2}."));
  PADDLE_ENFORCE_EQ(et3.type(),
                    phi::DataType::FLOAT32,
                    common::errors::InvalidArgument("Tensor data type should "
                                                    "be FLOAT32."));
  PADDLE_ENFORCE_EQ(et3.layout(),
                    phi::DataLayout::NCHW,
                    common::errors::InvalidArgument("Tensor layout should be "
                                                    "NCHW."));
  CHECK(phi::is_cpu_place(et3.place()));
  VLOG(6) << "Get impl";
  auto* dt3_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et3.impl())->data<float>();
  PADDLE_ENFORCE_EQ(dt3_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(dt3_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
  paddle::Tensor et4 = et3;
  VLOG(6) << "copy =";
  PADDLE_ENFORCE_EQ(et4.initialized(),
                    true,
                    common::errors::InvalidArgument("Tensor et4 should be "
                                                    "initialized."));
  auto* dt4_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et4.impl())->data<float>();
  PADDLE_ENFORCE_EQ(dt4_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(dt4_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
  VLOG(6) << "move =";
  paddle::Tensor et5 = std::move(et4);
  auto* dt5_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(et5.impl())->data<float>();
  PADDLE_ENFORCE_EQ(dt5_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(dt5_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
  VLOG(6) << "AutogradMeta";
  auto autograd_meta_test = std::make_shared<eager_test::AutogradMetaTest>(2);
  et3.set_autograd_meta(autograd_meta_test);
  auto* tmp_autograd_meta_test =
      static_cast<eager_test::AutogradMetaTest*>(et3.get_autograd_meta());
  PADDLE_ENFORCE_EQ(tmp_autograd_meta_test->val_,
                    2,
                    common::errors::InvalidArgument("AutogradMetaTest value "
                                                    "should be 2."));
}

TEST(EagerVariable, Constructor) {
  paddle::Tensor t3;
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, common::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(phi::CPUPlace());
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 10.0f;
  VLOG(6) << "Make Dense Tensor";
  t3.set_name("t3");
  VLOG(6) << "Set Name";
  PADDLE_ENFORCE_EQ(t3.name(),
                    "t3",
                    common::errors::InvalidArgument("Tensor name should be "
                                                    "'t3'."));
  PADDLE_ENFORCE_EQ(
      t3.defined(),
      false,
      common::errors::InvalidArgument(
          "Tensor t3 should be undefined but got %d.", t3.defined()));
  t3.set_impl(dt);

  egr::EagerVariable et3 = egr::EagerVariable(t3);
  VLOG(6) << "SyncToVar";
  PADDLE_ENFORCE_EQ(et3.Var().Get<phi::DenseTensor>().data<float>()[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(et3.Var().Get<phi::DenseTensor>().data<float>()[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
  VLOG(6) << "SyncToTensor";
  paddle::Tensor t4;
  t4.set_impl(et3.GetTensorBase());
  PADDLE_ENFORCE_EQ(
      t4.initialized(),
      true,
      common::errors::InvalidArgument(
          "Tensor t4 should be initialized but got %d.", t4.initialized()));

  VLOG(6) << "Check Tensor";
  auto* dt3_tmp_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(t4.impl())->data<float>();
  PADDLE_ENFORCE_EQ(dt3_tmp_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(dt3_tmp_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
  t4.reset();
  PADDLE_ENFORCE_EQ(
      t4.defined(),
      false,
      common::errors::InvalidArgument(
          "Tensor t4 should be undefined but got %d.", t4.defined()));

  VLOG(6) << "Check Tensor Copy_";
  std::vector<int64_t> rows = {1, 2};
  std::vector<int64_t> dims = {2};
  paddle::Tensor t7(std::make_shared<phi::SelectedRows>(rows, 2));
  std::dynamic_pointer_cast<phi::SelectedRows>(t7.impl())
      ->mutable_value()
      ->Resize(common::make_ddim(dims));
  auto* dt7_tmp_ptr = std::dynamic_pointer_cast<phi::SelectedRows>(t7.impl())
                          ->mutable_value()
                          ->mutable_data<float>(phi::CPUPlace());
  dt7_tmp_ptr[0] = 6.0f;
  dt7_tmp_ptr[1] = 11.0f;

  paddle::Tensor t8;
  paddle::Tensor t5;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  paddle::Tensor t6;
  paddle::Tensor t9;
  VLOG(6) << "Check Tensor Copy_ Selected Rows";
  t8.copy_(t7, phi::GPUPlace(0), false);
  t9.copy_(t8, phi::CPUPlace(), false);
  auto* dt9_tmp_ptr = std::dynamic_pointer_cast<phi::SelectedRows>(t9.impl())
                          ->value()
                          .data<float>();
  PADDLE_ENFORCE_EQ(dt9_tmp_ptr[0],
                    6.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "6.0f."));
  PADDLE_ENFORCE_EQ(dt9_tmp_ptr[1],
                    11.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "11.0f."));
  PADDLE_ENFORCE_EQ(
      std::dynamic_pointer_cast<phi::SelectedRows>(t9.impl())->height(),
      2,
      common::errors::InvalidArgument("SelectedRows height should "
                                      "be 2."));

  VLOG(6) << "Check Tensor Copy_ Dense Tensor";
  t5.copy_(t3, phi::GPUPlace(0), false);
  t6.copy_(t5, phi::CPUPlace(), false);
  auto* dt6_tmp_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(t6.impl())->data<float>();
  PADDLE_ENFORCE_EQ(dt6_tmp_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(dt6_tmp_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
#else
  t5.copy_(t3, phi::CPUPlace(), false);
  auto* dt5_tmp_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(t5.impl())->data<float>();
  PADDLE_ENFORCE_EQ(dt5_tmp_ptr[0],
                    5.0f,
                    common::errors::InvalidArgument("First element should be "
                                                    "5.0f."));
  PADDLE_ENFORCE_EQ(dt5_tmp_ptr[1],
                    10.0f,
                    common::errors::InvalidArgument("Second element should be "
                                                    "10.0f."));
#endif

  VLOG(6) << "Finish";
}

TEST(EagerVariable, DataLayout) {
  paddle::Tensor tensor;
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           common::make_ddim({1, 1, 1, 1}),
                           phi::DataLayout::UNDEFINED);
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(phi::CPUPlace());
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 5.0f;
  dt_ptr[2] = 5.0f;
  dt_ptr[3] = 5.0f;
  tensor.set_impl(dt);
  auto eager_var = std::make_shared<egr::EagerVariable>(tensor);
  auto layout = paddle::imperative::GetDataLayout(eager_var);
  PADDLE_ENFORCE_EQ(layout,
                    phi::DataLayout::UNDEFINED,
                    common::errors::InvalidArgument("Data layout should be "
                                                    "UNDEFINED."));
  paddle::imperative::SetDataLayout(eager_var, phi::DataLayout::NCHW);
  layout = paddle::imperative::GetDataLayout(eager_var);
  PADDLE_ENFORCE_EQ(layout,
                    phi::DataLayout::NCHW,
                    common::errors::InvalidArgument("Data layout should be "
                                                    "NCHW."));
}

TEST(VariableCompatTensor, MemberFunction) {
  egr::VariableCompatTensor var_tensor;
  // test GetMutable and Get
  var_tensor.GetMutable<phi::Vocab>();
  auto& vocab = var_tensor.Get<phi::Vocab>();
  EXPECT_EQ(vocab.size(), 0UL);
  bool caught_exception = false;
  try {
    var_tensor.GetMutable<phi::Strings>();
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("The Variable type must be") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
  // test Type and IsType
  EXPECT_TRUE(var_tensor.IsType<phi::Vocab>());
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
