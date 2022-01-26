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

#include "paddle/pten/api/include/api.h"

#include "paddle/pten/api/include/sparse_utils.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/sparse_coo_tensor.h"

TEST(API, to_sparse_coo) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 2;
  auto* dense_x_data = dense_x->mutable_data<float>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  const int64_t non_zero_num = 4;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  pten::CPUContext dev_ctx_cpu;

  // 1. test cpu
  paddle::experimental::Tensor x(dense_x);
  auto out =
      paddle::experimental::to_sparse_coo(x, pten::Backend::CPU, sparse_dim);
  auto coo = std::dynamic_pointer_cast<pten::SparseCooTensor>(out.impl());
  ASSERT_EQ(coo->nnz(), non_zero_num);
  int cmp_indices = memcmp(coo->non_zero_indices().data<int64_t>(),
                           indices_data.data(),
                           indices_data.size() * sizeof(int64_t));
  ASSERT_EQ(cmp_indices, 0);
  int cmp_elements = memcmp(coo->non_zero_elements().data<float>(),
                            non_zero_data.data(),
                            non_zero_data.size() * sizeof(float));
  ASSERT_EQ(cmp_elements, 0);
}
