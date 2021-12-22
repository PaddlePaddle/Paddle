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

#include "gtest/gtest.h"

#include "paddle/pten/api/include/sparse_coo_tensor_utils.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/sparse_coo_tensor.h"

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, to_sparse_coo) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();
  float sparse_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> dense_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<std::vector<int64_t>> indices_data = {
      {0, 1}, {1, 0}, {1, 2}, {2, 0}};

  std::copy(&sparse_data[0][0], &sparse_data[0][0] + 9, dense_x_data);

  paddle::experimental::Tensor x(dense_x);
  const int64_t sparse_dim = 2;
  auto out = paddle::experimental::to_sparse_coo(x, sparse_dim);
  auto sparse_out =
      std::dynamic_pointer_cast<pten::SparseCooTensor>(out.impl());
  const auto& values = sparse_out->non_zero_elements();
  int64_t non_zero_num = values.numel();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), dense_data.size());

  const auto& indices = sparse_out->non_zero_indices();
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(values.data<float>()[i], dense_data[i]);
  }
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(indices.data<int64_t>()[i], indices_data[i][0]);
    ASSERT_EQ(indices.data<int64_t>()[non_zero_num + i], indices_data[i][1]);
  }
}

TEST(API, to_sparse_coo_hybird) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({5, 2}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();
  float sparse_data[5][2] = {
      {0.0, 1.0}, {0.0, 0.0}, {2.0, 0.0}, {3.0, 3.2}, {0.0, 0.0}};
  std::vector<std::vector<float>> dense_data = {
      {0.0, 1.0}, {2.0, 0.0}, {3.0, 3.2}};
  std::vector<int64_t> indices_data = {0, 2, 3};

  std::copy(&sparse_data[0][0], &sparse_data[0][0] + 10, dense_x_data);

  paddle::experimental::Tensor x(dense_x);
  const int64_t sparse_dim = 1;
  auto out = paddle::experimental::to_sparse_coo(x, sparse_dim);
  auto sparse_out =
      std::dynamic_pointer_cast<pten::SparseCooTensor>(out.impl());
  const auto& values = sparse_out->non_zero_elements();
  const auto& indices = sparse_out->non_zero_indices();
  int64_t non_zero_num = indices.numel();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), dense_data.size());
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(values.data<float>()[i * 2], dense_data[i][0]);
    ASSERT_EQ(values.data<float>()[i * 2 + 1], dense_data[i][1]);
  }
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(indices.data<int64_t>()[i], indices_data[i]);
  }
}
}  // namespace tests
}  // namespace paddle
