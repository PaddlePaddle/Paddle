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

#include "paddle/pten/api/include/sparse_csr_tensor_utils.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/sparse_csr_tensor.h"
#include "paddle/pten/kernels/gpu/utils.h"

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, to_sparse_csr) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::experimental::Tensor x(dense_x);
  auto out = paddle::experimental::to_sparse_csr(x, pten::Backend::CPU);
  auto sparse_out =
      std::dynamic_pointer_cast<pten::SparseCsrTensor>(out.impl());
  int64_t non_zero_num = sparse_out->nnz();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), non_zero_data.size());

  const auto& crows = sparse_out->non_zero_crows();
  const auto& cols = sparse_out->non_zero_cols();
  const auto& non_zero_elements = sparse_out->non_zero_elements();
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
    ASSERT_EQ(cols.data<int64_t>()[i], cols_data[i]);
  }
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(crows.data<int64_t>()[i], crows_data[i]);
  }
}

TEST(API, to_sparse_csr_cuda) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const auto alloc_cuda =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));
  auto d_dense_x = std::make_shared<pten::DenseTensor>(
      alloc_cuda,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));

  auto* dense_x_data = dense_x->mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx =
      static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda);
  pten::Copy(*dev_ctx, *dense_x, true, d_dense_x.get());

  paddle::experimental::Tensor x(d_dense_x);
  auto out = paddle::experimental::to_sparse_csr(x, pten::Backend::GPU);
  auto sparse_out =
      std::dynamic_pointer_cast<pten::SparseCsrTensor>(out.impl());
  int64_t non_zero_num = sparse_out->nnz();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), non_zero_data.size());

  // const auto& crows = sparse_out->non_zero_crows();
  // const auto& cols = sparse_out->non_zero_cols();
  // const auto& non_zero_elements = sparse_out->non_zero_elements();
  // for (int64_t i = 0; i < non_zero_num; i++) {
  //  ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
  //  ASSERT_EQ(cols.data<int64_t>()[i], cols_data[i]);
  //}
  // for (int i = 0; i < 4; i++) {
  //  ASSERT_EQ(crows.data<int64_t>()[i], crows_data[i]);
  //}
}
}  // namespace tests
}  // namespace paddle
