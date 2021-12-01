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
  std::vector<float> sparse_data = {
      0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.0, 0.0, 0.0};
  std::vector<float> dense_data = {1.0, 2.0, 3.0, 3.0};

  std::copy(sparse_data.begin(), sparse_data.end(), dense_x_data);

  paddle::experimental::Tensor x(dense_x);
  const int64_t sparse_dim = 2;
  auto out = paddle::experimental::to_sparse_coo(x, sparse_dim);
  auto sparse_out =
      std::dynamic_pointer_cast<pten::SparseCooTensor>(out.impl());
  const auto& values = sparse_out->values();
  for (size_t i = 0; i < dense_data.size(); i++) {
    ASSERT_EQ(values.data<float>()[i], dense_data[i]);
  }
}

}  // namespace tests
}  // namespace paddle
