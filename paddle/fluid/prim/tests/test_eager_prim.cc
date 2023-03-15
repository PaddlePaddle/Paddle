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

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/prim/tests/init_env_utils.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"

DECLARE_string(tensor_operants_mode);

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(pow, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_and, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_or, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_xor, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_not, CPU, ALL_LAYOUT);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(full, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh_grad, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(pow, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_and, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_or, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_xor, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(bitwise_not, KPS, ALL_LAYOUT);

#endif

namespace paddle {
namespace prim {

TEST(EagerPrim, TanhBackwardTest) {
  // 1. Initialized
  eager_test::InitEnv(paddle::platform::CPUPlace());
  FLAGS_tensor_operants_mode = "eager";
  paddle::prim::InitTensorOperants();
  // 2. pre
  paddle::framework::DDim ddim = phi::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor0 =
      ::egr::egr_utils_api::CreateTensorWithValue(ddim,
                                                  paddle::platform::CPUPlace(),
                                                  phi::DataType::FLOAT32,
                                                  phi::DataLayout::NCHW,
                                                  5.0 /*value*/,
                                                  true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor0);
  paddle::Tensor tensor1 =
      ::egr::egr_utils_api::CreateTensorWithValue(ddim,
                                                  paddle::platform::CPUPlace(),
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
  eager_test::InitEnv(paddle::platform::CPUPlace());
  FLAGS_tensor_operants_mode = "eager";
  paddle::prim::InitTensorOperants();
  // 2. pre
  paddle::framework::DDim ddim = phi::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor0 =
      ::egr::egr_utils_api::CreateTensorWithValue(ddim,
                                                  paddle::platform::CPUPlace(),
                                                  phi::DataType::INT32,
                                                  phi::DataLayout::NCHW,
                                                  1 /*value*/,
                                                  true /*is_leaf*/);
  ::egr::egr_utils_api::RetainGradForTensor(tensor0);
  paddle::Tensor tensor1 =
      ::egr::egr_utils_api::CreateTensorWithValue(ddim,
                                                  paddle::platform::CPUPlace(),
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

TEST(EagerPrim, TestFlags) {
  PrimCommonUtils::SetBwdPrimEnabled(true);
  ASSERT_TRUE(PrimCommonUtils::IsBwdPrimEnabled());
  PrimCommonUtils::SetBwdPrimEnabled(false);
  ASSERT_FALSE(PrimCommonUtils::IsBwdPrimEnabled());
}

}  // namespace prim
}  // namespace paddle
