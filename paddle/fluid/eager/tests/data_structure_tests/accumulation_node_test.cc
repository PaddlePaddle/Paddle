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
#include "paddle/pten/api/lib/utils/allocator.h"

#include "paddle/pten/core/kernel_registry.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(AccumulationNode, EagerTensor) {
  // Construct Eager Tensor
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT16, paddle::framework::make_ddim({1, 1}));
  std::shared_ptr<pten::DenseTensor> dt0 = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  dt0->mutable_data<paddle::platform::float16>()[0] = 10.0;
  EagerTensor et0 = EagerTensor(dt0);

  std::shared_ptr<pten::DenseTensor> dt1 = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);

  dt1->mutable_data<paddle::platform::float16>()[0] = 20.0;
  EagerTensor et1 = EagerTensor(dt1);

  std::shared_ptr<pten::DenseTensor> grad_dt =
      std::make_shared<pten::DenseTensor>(
          std::make_shared<paddle::experimental::DefaultAllocator>(
              paddle::platform::CPUPlace()),
          meta);
  EagerTensor grad_et = EagerTensor(grad_dt);

  // AccumulationNode
  GradNodeAccumulation node = GradNodeAccumulation();

  // Hook
  std::function<egr::EagerTensor(const egr::EagerTensor&)> hook =
      [&grad_et](const egr::EagerTensor& t) {
        if (t.defined()) {
          grad_et.set_impl(t.impl());
          return grad_et;
        } else {
          grad_et.MutableVar()
              ->GetMutable<paddle::framework::LoDTensor>()
              ->ShareDataWith(t.Var().Get<paddle::framework::LoDTensor>());
          return grad_et;
        }
      };
  node.RetainGrad(hook);

  // operator()
  EagerTensor ret_et0 = node({{et0}})[0][0];
  auto* ret_et0_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(ret_et0.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et0_ptr[0], paddle::platform::float16(10.0f));

  EagerTensor ret_et1 = node({{et1}})[0][0];
  auto* ret_et1_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(ret_et1.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_et1_ptr[0], paddle::platform::float16(30.0f));

  // Retain Grad
  auto* ret_grad_et_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_et.impl())
          ->data<paddle::platform::float16>();
  CHECK_EQ(ret_grad_et_ptr[0], paddle::platform::float16(30.0f));
}
