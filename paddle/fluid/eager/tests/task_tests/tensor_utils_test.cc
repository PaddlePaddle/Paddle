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

#include <sstream>

#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/phi/api/lib/utils/allocator.h"

#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace egr {

TEST(TensorUtils, Test) {
  // Prepare Device Contexts
  eager_test::InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::experimental::Tensor> target_tensors;
  paddle::framework::DDim ddim = phi::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::experimental::Tensor t = egr_utils_api::CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 5.0 /*value*/, true /*is_leaf*/);

  paddle::experimental::Tensor t_grad = egr_utils_api::CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 1.0 /*value*/, false /*is_leaf*/);

  CHECK_EQ(egr_utils_api::IsLeafTensor(t), true);

  // Test Utils
  eager_test::CompareTensorWithValue<float>(t, 5.0);

  egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(&t);
  *meta->MutableGrad() = t_grad;

  eager_test::CompareGradTensorWithValue<float>(t, 1.0);
}

}  // namespace egr
