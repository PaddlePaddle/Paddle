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

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/kernel_registry.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(AccumulationNode, Tensor) {
  // Construct Eager Tensor
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT16, paddle::framework::make_ddim({1, 1}));
  std::shared_ptr<pten::DenseTensor> dt0 = std::make_shared<pten::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  dt0->mutable_data<paddle::platform::float16>(
      paddle::platform::CPUPlace())[0] = 10.0;
  paddle::experimental::Tensor et0 = paddle::experimental::Tensor(dt0);

  std::shared_ptr<pten::DenseTensor> dt1 = std::make_shared<pten::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);

  dt1->mutable_data<paddle::platform::float16>(
      paddle::platform::CPUPlace())[0] = 20.0;
  paddle::experimental::Tensor et1 = paddle::experimental::Tensor(dt1);

  std::shared_ptr<pten::DenseTensor> grad_dt =
      std::make_shared<pten::DenseTensor>(
          std::make_unique<paddle::experimental::DefaultAllocator>(
              paddle::platform::CPUPlace())
              .get(),
          meta);
  paddle::experimental::Tensor grad_et = paddle::experimental::Tensor(grad_dt);
  auto grad_meta = EagerUtils::autograd_meta(&grad_et);

  // AccumulationNode
  auto node = std::make_shared<GradNodeAccumulation>(grad_meta);
  grad_meta->SetGradNode(node);
  grad_meta->SetStopGradient(false);

  egr::egr_utils_api::RetainGradForTensor(grad_et);

  // operator()
  paddle::experimental::Tensor ret_et0 = node->operator()({{et0}})[0][0];
  auto* ret_et0_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(ret_et0.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et0_ptr[0], paddle::platform::float16(10.0f));

  paddle::experimental::Tensor ret_et1 = node->operator()({{et1}})[0][0];
  auto* ret_et1_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(ret_et1.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et1_ptr[0], paddle::platform::float16(20.0f));

  // Check Retain Grad
  paddle::experimental::Tensor* grad = EagerUtils::mutable_grad(grad_et);
  auto* grad_ptr = std::dynamic_pointer_cast<pten::DenseTensor>(grad->impl())
                       ->data<paddle::platform::float16>();
  CHECK_EQ(grad_ptr[0], paddle::platform::float16(30.0f));
}
