// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "test/cpp/eager/test_utils.h"
#include "test/cpp/prim/init_env_utils.h"

COMMON_DECLARE_string(tensor_operants_mode);

namespace paddle {
namespace prim {

TEST(EagerPrim, TanhBackwardTest) {
  // 1. Initialized
  eager_test::InitEnv(phi::CPUPlace());
  FLAGS_tensor_operants_mode = "eager";
  paddle::prim::InitTensorOperants();
  // 2. pre
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor0 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor0);
  paddle::Tensor tensor1 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor1);
  // 3. Run Forward once
  paddle::Tensor out0 = tanh_ad_func(tensor0);
  std::vector<paddle::Tensor> outs0 = {out0};
  // Disable prim
  PrimCommonUtils::SetBwdPrimEnabled(false);
  ASSERT_FALSE(PrimCommonUtils::IsBwdPrimEnabled());
  // 4. Run Backward
  egr::Backward(outs0, {}, false);

  paddle::Tensor out1 = tanh_ad_func(tensor1);
  std::vector<paddle::Tensor> outs1 = {out1};
  // Enable prim
  PrimCommonUtils::SetBwdPrimEnabled(true);
  ASSERT_TRUE(PrimCommonUtils::IsBwdPrimEnabled());
  // 4. Run Backward
  ::egr::Backward(outs1, {}, false);
  VLOG(7)
      << "Target Grad is: "
      << std::static_pointer_cast<phi::DenseTensor>(
             ::egr::EagerUtils::unsafe_autograd_meta(tensor0)->Grad().impl())
             ->data<float>()[0];
  VLOG(7)
      << "Result Grad is: "
      << std::static_pointer_cast<phi::DenseTensor>(
             ::egr::EagerUtils::unsafe_autograd_meta(tensor1)->Grad().impl())
             ->data<float>()[0];
  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(
      tensor1,
      std::static_pointer_cast<phi::DenseTensor>(
          ::egr::EagerUtils::unsafe_autograd_meta(tensor0)->Grad().impl())
          ->data<float>()[0]);
}

TEST(EagerPrim, LogicalOperantsTest) {
  // 1. Initialized
  eager_test::InitEnv(phi::CPUPlace());
  FLAGS_tensor_operants_mode = "eager";
  paddle::prim::InitTensorOperants();
  // 2. pre
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor0 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::INT32,
                                        phi::DataLayout::NCHW,
                                        1 /*value*/,
                                        true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor0);
  paddle::Tensor tensor1 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::INT32,
                                        phi::DataLayout::NCHW,
                                        0 /*value*/,
                                        true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor1);
  // 3. Run Forward once
  paddle::Tensor out0 = tensor0 & tensor1;
  paddle::Tensor out1 = bitwise_and_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<int>()[0], out1.data<int>()[0]);
  out0 = tensor0 | tensor1;
  out1 = bitwise_or_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<int>()[0], out1.data<int>()[0]);
  out0 = tensor0 ^ tensor1;
  out1 = bitwise_xor_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<int>()[0], out1.data<int>()[0]);
  out0 = ~tensor0;
  out1 = bitwise_not_ad_func(tensor0);
  EXPECT_EQ(out0.data<int>()[0], out1.data<int>()[0]);
}

TEST(EagerPrim, CompareOperantsTest) {
  // 1. Initialized
  eager_test::InitEnv(phi::CPUPlace());
  FLAGS_tensor_operants_mode = "eager";
  paddle::prim::InitTensorOperants();
  // 2. pre
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor0 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::INT32,
                                        phi::DataLayout::NCHW,
                                        1 /*value*/,
                                        true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor0);
  paddle::Tensor tensor1 =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::INT32,
                                        phi::DataLayout::NCHW,
                                        0 /*value*/,
                                        true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor1);
  // 3. Run Forward once
  paddle::Tensor out0 = (tensor0 < tensor1);
  paddle::Tensor out1 = less_than_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<bool>()[0], out1.data<bool>()[0]);
  out0 = (tensor0 <= tensor1);
  out1 = less_equal_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<bool>()[0], out1.data<bool>()[0]);
  out0 = (tensor0 == tensor1);
  out1 = equal_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<bool>()[0], out1.data<bool>()[0]);
  out0 = (tensor0 != tensor1);
  out1 = not_equal_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<bool>()[0], out1.data<bool>()[0]);
  out0 = (tensor0 > tensor1);
  out1 = greater_than_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<bool>()[0], out1.data<bool>()[0]);
  out0 = (tensor0 >= tensor1);
  out1 = greater_equal_ad_func(tensor0, tensor1);
  EXPECT_EQ(out0.data<bool>()[0], out1.data<bool>()[0]);
}

TEST(EagerPrim, TestFlags) {
  PrimCommonUtils::SetBwdPrimEnabled(true);
  ASSERT_TRUE(PrimCommonUtils::IsBwdPrimEnabled());
  PrimCommonUtils::SetBwdPrimEnabled(false);
  ASSERT_FALSE(PrimCommonUtils::IsBwdPrimEnabled());
}

}  // namespace prim
}  // namespace paddle
