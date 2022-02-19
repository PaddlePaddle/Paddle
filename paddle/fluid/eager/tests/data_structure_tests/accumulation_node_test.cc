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
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/phi/api/lib/utils/allocator.h"

#include "paddle/phi/core/kernel_registry.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(AccumulationNode, Tensor) {
  // Construct Eager Tensor
  pten::DenseTensorMeta meta =
      pten::DenseTensorMeta(pten::DataType::FLOAT16, pten::make_ddim({1, 1}));
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

  // AccumulationNode
  GradNodeAccumulation node = GradNodeAccumulation();

  // Hook, RetainGrad
  std::function<paddle::experimental::Tensor(
      const paddle::experimental::Tensor&)>
      hook = [&grad_et](const paddle::experimental::Tensor& t) {
        grad_et.set_impl(t.impl());
        return grad_et;
      };
  node.RetainGrad(hook);

  // operator()
  paddle::experimental::Tensor ret_et0 = node({{et0}})[0][0];
  auto* ret_et0_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(ret_et0.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et0_ptr[0], paddle::platform::float16(10.0f));

  paddle::experimental::Tensor ret_et1 = node({{et1}})[0][0];
  auto* ret_et1_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(ret_et1.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et1_ptr[0], paddle::platform::float16(30.0f));

  // Retain Grad
  auto* ret_grad_et_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_et.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_grad_et_ptr[0], paddle::platform::float16(30.0f));

  // Reduce Hook case 1: Call RegisterReduceHook and run operator()
  VLOG(6) << "Test Reduce Hook";
  auto reduce_hook_1 = [&](void) -> void {
    auto* grad_et_ptr =
        std::dynamic_pointer_cast<pten::DenseTensor>(grad_et.impl())
            ->data<paddle::platform::float16>();
    grad_et_ptr[0] = 36.0;
    VLOG(6) << "Running Reduce Hook";
  };

  node.RegisterReduceHook(reduce_hook_1);

  // operator()
  paddle::experimental::Tensor _ret = node({{et0}})[0][0];

  // Check operator() result, should be 36.0
  auto* _ret_ptr = std::dynamic_pointer_cast<pten::DenseTensor>(_ret.impl())
                       ->data<paddle::platform::float16>();
  CHECK_EQ(_ret_ptr[0], paddle::platform::float16(36.0f));

  // Check Retain Grad, should be 36.0
  auto* _ret_grad_et_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_et.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(_ret_grad_et_ptr[0], paddle::platform::float16(36.0f));

  // Reduce Hook case 2: Call RegisterReduceHook and ApplyReduceHooks directly
  VLOG(6) << "Test Reduce Hook";
  auto reduce_hook_2 = [&](void) -> void {
    auto* ret_et0_ptr = std::dynamic_pointer_cast<pten::DenseTensor>(et0.impl())
                            ->data<paddle::platform::float16>();
    ret_et0_ptr[0] = 100.0;  // set to 100.0
    VLOG(6) << "Running Reduce Hook";
  };
  node.RegisterReduceHook(reduce_hook_2);
  node.ApplyReduceHooks();

  // Check ApplyReduceHooks result
  CHECK_EQ(std::dynamic_pointer_cast<pten::DenseTensor>(et0.impl())
               ->data<paddle::platform::float16>()[0],
           paddle::platform::float16(100.0f));
}
