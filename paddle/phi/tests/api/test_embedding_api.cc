/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include <memory>

#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(sparse_weight_embedding, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_weight_embedding_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_weight_embedding_sparse_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(empty, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

TEST(API, sparse_weight_embedding) {
  auto x = paddle::experimental::empty({4}, DataType::INT32);
  auto* x_data = x.data<int32_t>();
  x_data[0] = 0;
  x_data[1] = 4;
  x_data[2] = 3;
  x_data[3] = 1;

  auto weight_sr = std::make_shared<phi::SelectedRows>(
      std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6}, 16);
  *weight_sr->mutable_value() = *static_cast<phi::DenseTensor*>(
      paddle::experimental::full({7, 3}, 2, DataType::FLOAT32).impl().get());
  paddle::experimental::Tensor weight;
  weight.set_impl(weight_sr);

  auto out = paddle::experimental::embedding(x, weight);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 4);
  ASSERT_EQ(out.numel(), 12);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
}

TEST(API, sparse_weight_embedding_grad) {
  auto x = paddle::experimental::empty({4}, DataType::INT32);
  auto* x_data = x.data<int32_t>();
  x_data[0] = 0;
  x_data[1] = 4;
  x_data[2] = 3;
  x_data[3] = 1;

  auto weight_sr = std::make_shared<phi::SelectedRows>(
      std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6}, 16);
  *weight_sr->mutable_value() = *static_cast<phi::DenseTensor*>(
      paddle::experimental::full({7, 3}, 2, DataType::FLOAT32).impl().get());
  paddle::experimental::Tensor weight;
  weight.set_impl(weight_sr);

  auto out_grad = paddle::experimental::full({4, 3}, 1, DataType::FLOAT32);

  paddle::experimental::Tensor weight_grad;

  paddle::experimental::embedding_grad(
      x, weight, out_grad, -1, false, &weight_grad);

  // 3. check result
  ASSERT_EQ(weight_grad.dims().size(), 2);
  ASSERT_EQ(weight_grad.dims()[0], 16);
  ASSERT_EQ(weight_grad.numel(), 48);
  ASSERT_EQ(weight_grad.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(weight_grad.layout(), phi::DataLayout::NCHW);
}

TEST(API, sparse_weight_embedding_sparse_grad) {
  auto x = paddle::experimental::empty({4}, DataType::INT32);
  auto* x_data = x.data<int32_t>();
  x_data[0] = 0;
  x_data[1] = 4;
  x_data[2] = 3;
  x_data[3] = 1;

  auto weight_sr = std::make_shared<phi::SelectedRows>(
      std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6}, 16);
  *weight_sr->mutable_value() = *static_cast<phi::DenseTensor*>(
      paddle::experimental::full({7, 3}, 2, DataType::FLOAT32).impl().get());
  paddle::experimental::Tensor weight;
  weight.set_impl(weight_sr);

  auto out_grad = paddle::experimental::full({4, 3}, 1, DataType::FLOAT32);

  paddle::experimental::Tensor weight_grad;

  paddle::experimental::embedding_grad(
      x, weight, out_grad, -1, true, &weight_grad);

  // 3. check result
  ASSERT_EQ(weight_grad.dims().size(), 2);
  ASSERT_EQ(weight_grad.dims()[0], 4);
  ASSERT_EQ(weight_grad.numel(), 12);
  ASSERT_EQ(weight_grad.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(weight_grad.layout(), phi::DataLayout::NCHW);
}

}  // namespace tests
}  // namespace paddle
