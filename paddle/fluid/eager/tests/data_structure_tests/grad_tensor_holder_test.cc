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

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/pten/api/lib/utils/allocator.h"

#include "paddle/pten/core/kernel_registry.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(GradTensorHolder, Constructor) {
  GradSlotMeta slot_meta;
  slot_meta.Init(1);
  GradTensorHolder grad_tensor_holder = GradTensorHolder({slot_meta});
  GradTensorHolder grad_tensor_holder2 = GradTensorHolder(grad_tensor_holder);

  // Construct Eager Tensor
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({2, 2}));
  std::shared_ptr<pten::DenseTensor> dt = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  EagerTensor et = EagerTensor(dt);

  std::vector<std::vector<EagerTensor>> inputs;
  inputs.push_back({et});

  GradTensorHolder grad_tensor_holder4 = GradTensorHolder(std::move(inputs));
}

TEST(GradTensorHolder, Interfaces) {
  // Construct Eager Tensor
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({1, 1}));
  std::shared_ptr<pten::DenseTensor> dt0 = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  dt0->mutable_data<float>()[0] = 10.0;
  EagerTensor et0 = EagerTensor(dt0);

  std::shared_ptr<pten::DenseTensor> dt1 = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  dt1->mutable_data<float>()[0] = 20.0;
  EagerTensor et1 = EagerTensor(dt1);

  // Constructor empty GradTensorHolder
  GradSlotMeta slot_meta;
  slot_meta.Init(1);
  GradTensorHolder grad_tensor_holder =
      GradTensorHolder({slot_meta, slot_meta});

  // add():
  // fill one
  grad_tensor_holder.add(0, 0, et0, true);

  // accumulation
  grad_tensor_holder.add(1, 0, et0, false);
  grad_tensor_holder.add(1, 0, et1, false);

  // Buffers()
  const auto& buffers = grad_tensor_holder.Buffers();
  CHECK_EQ(static_cast<int>(buffers.size()), 2);
  CHECK_EQ(static_cast<int>(buffers[0].size()), 1);
  CHECK_EQ(static_cast<int>(buffers[1].size()), 1);

  // operator[]
  const auto& holder_et0 = grad_tensor_holder[0][0];
  const auto& holder_et1 = grad_tensor_holder[1][0];

  auto* holder_et0_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(holder_et0.impl())
          ->data<float>();
  auto* holder_et1_ptr =
      std::dynamic_pointer_cast<pten::DenseTensor>(holder_et1.impl())
          ->data<float>();

  CHECK_EQ(holder_et0_ptr[0], 1.0f);
  CHECK_EQ(holder_et1_ptr[0], 30.0f);
}
