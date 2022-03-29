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

#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(AccumulationNode, Tensor) {
  // Construct Eager Tensor
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT16, phi::make_ddim({1, 1}));
  std::shared_ptr<phi::DenseTensor> dt0 = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);
  dt0->mutable_data<paddle::platform::float16>(
      paddle::platform::CPUPlace())[0] = paddle::platform::float16(10.0f);
  paddle::experimental::Tensor et0 = paddle::experimental::Tensor(dt0);

  std::shared_ptr<phi::DenseTensor> dt1 = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace())
          .get(),
      meta);

  dt1->mutable_data<paddle::platform::float16>(
      paddle::platform::CPUPlace())[0] = paddle::platform::float16(20.0f);
  paddle::experimental::Tensor et1 = paddle::experimental::Tensor(dt1);

  std::shared_ptr<phi::DenseTensor> input_dt =
      std::make_shared<phi::DenseTensor>(
          std::make_unique<paddle::experimental::DefaultAllocator>(
              paddle::platform::CPUPlace())
              .get(),
          meta);
  paddle::experimental::Tensor input_et =
      paddle::experimental::Tensor(input_dt);
  auto grad_meta = EagerUtils::autograd_meta(&input_et);

  // Initialize Grad Tensor
  std::shared_ptr<phi::DenseTensor> grad_dt =
      std::make_shared<phi::DenseTensor>(
          std::make_unique<paddle::experimental::DefaultAllocator>(
              paddle::platform::CPUPlace())
              .get(),
          meta);
  grad_dt->mutable_data<paddle::platform::float16>(
      paddle::platform::CPUPlace())[0] = paddle::platform::float16(0.0f);
  grad_meta->MutableGrad()->set_impl(grad_dt);

  // AccumulationNode
  auto node = std::make_shared<GradNodeAccumulation>(grad_meta);
  grad_meta->SetGradNode(node);
  grad_meta->SetStopGradient(false);

  // operator()
  std::vector<std::vector<paddle::experimental::Tensor>> et0_vec = {{et0}};
  paddle::experimental::Tensor ret_et0 = node->operator()(et0_vec)[0][0];
  auto* ret_et0_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(ret_et0.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et0_ptr[0], paddle::platform::float16(10.0f));

  std::vector<std::vector<paddle::experimental::Tensor>> et1_vec = {{et1}};
  paddle::experimental::Tensor ret_et1 = node->operator()(et1_vec)[0][0];

  auto* ret_et1_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(ret_et1.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et1_ptr[0], paddle::platform::float16(20.0f));

  // Check Retain Grad
  CHECK_EQ(std::dynamic_pointer_cast<phi::DenseTensor>(et0.impl())
               ->data<paddle::platform::float16>()[0],
           paddle::platform::float16(10.0f));
  paddle::experimental::Tensor* grad = EagerUtils::mutable_grad(input_et);
  auto* grad_ptr = std::dynamic_pointer_cast<phi::DenseTensor>(grad->impl())
                       ->data<paddle::platform::float16>();
  CHECK_EQ(grad_ptr[0], paddle::platform::float16(30.0f));

  // Reduce Hook case 1: Call RegisterReduceHook and run operator()
  VLOG(6) << "Test Reduce Hook";
  CHECK_EQ(std::dynamic_pointer_cast<phi::DenseTensor>(et0.impl())
               ->data<paddle::platform::float16>()[0],
           paddle::platform::float16(10.0f));

  auto reduce_hook_1 = [&](void) -> void {
    auto* input_et_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(input_et.impl())
            ->mutable_data<paddle::platform::float16>(
                paddle::platform::CPUPlace());
    input_et_ptr[0] = 36.0;
    VLOG(6) << "Running Reduce Hook";
  };

  node->RegisterReduceHook(
      std::make_shared<egr::CppTensorVoidHook>(reduce_hook_1));

  // operator()
  paddle::experimental::Tensor _ret = node->operator()(et0_vec)[0][0];

  // Check operator() result, should be 36.0
  auto* _ret_ptr = std::dynamic_pointer_cast<phi::DenseTensor>(_ret.impl())
                       ->data<paddle::platform::float16>();
  CHECK_EQ(_ret_ptr[0], paddle::platform::float16(10.0f));

  // Check Retain Grad, should be 36.0
  auto* _ret_input_et_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(input_et.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(_ret_input_et_ptr[0], paddle::platform::float16(36.0f));

  // Reduce Hook case 2: Call RegisterReduceHook and ApplyReduceHooks directly
  VLOG(6) << "Test Reduce Hook";
  auto reduce_hook_2 = [&](void) -> void {
    auto* ret_et0_ptr = std::dynamic_pointer_cast<phi::DenseTensor>(et0.impl())
                            ->mutable_data<paddle::platform::float16>(
                                paddle::platform::CPUPlace());
    ret_et0_ptr[0] = 100.0;  // set to 100.0
    VLOG(6) << "Running Reduce Hook";
  };
  node->RegisterReduceHook(
      std::make_shared<egr::CppTensorVoidHook>(reduce_hook_2));
  node->ApplyReduceHooks();

  // Check ApplyReduceHooks result
  CHECK_EQ(std::dynamic_pointer_cast<phi::DenseTensor>(et0.impl())
               ->data<paddle::platform::float16>()[0],
           paddle::platform::float16(100.0f));
}
