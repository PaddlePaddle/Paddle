/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"

#include "paddle/pten/api/lib/utils/tensor_utils.h"

namespace paddle {
namespace tests {

using DDim = paddle::framework::DDim;
using DataType = paddle::experimental::DataType;
using DataLayout = paddle::experimental::DataLayout;

using DenseTensor = pten::DenseTensor;
using DenseTensorMeta = pten::DenseTensorMeta;

TEST(tensor_utils, dense_tensor_to_lod_tensor) {
  const DDim dims({2, 1});
  const DataType dtype{DataType::FLOAT32};
  const DataLayout layout{DataLayout::NCHW};
  const std::vector<std::vector<size_t>> lod{{0, 2}};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  auto alloc =
      std::make_shared<experimental::DefaultAllocator>(platform::CPUPlace());

  DenseTensor dense_tensor(alloc, meta);
  float* data = dense_tensor.mutable_data<float>();
  data[0] = 1.0f;
  data[1] = 2.1f;

  framework::LoDTensor lod_tensor;
  experimental::MovesStorage(&dense_tensor, &lod_tensor);

  CHECK(dense_tensor.lod().size() == lod_tensor.lod().size());
  CHECK(dense_tensor.lod()[0] ==
        static_cast<std::vector<size_t>>((lod_tensor.lod()[0])));
  CHECK(dense_tensor.dtype() == pten::TransToPtenDataType(lod_tensor.type()));
  CHECK(dense_tensor.layout() ==
        pten::TransToPtenDataLayout(lod_tensor.layout()));
  CHECK(platform::is_cpu_place(lod_tensor.place()));

  CHECK(lod_tensor.data<float>()[0] == 1.0f);
  CHECK(lod_tensor.data<float>()[1] == 2.1f);

  auto dense_tensor_1 = experimental::MakePtenDenseTensor(lod_tensor);
  CHECK(dense_tensor_1->dims() == dims);
  CHECK(dense_tensor_1->dtype() == dtype);
  CHECK(dense_tensor_1->layout() == layout);
  CHECK(dense_tensor_1->lod().size() == lod.size());
  CHECK(dense_tensor_1->lod()[0] == lod[0]);
  const float* data_1 = dense_tensor_1->data<float>();
  CHECK(data_1[0] == 1.0f);
  CHECK(data_1[1] == 2.1f);
}

TEST(tensor_utils, dense_tensor_to_tensor) {
  const DDim dims({2, 1});
  const DataType dtype{DataType::FLOAT32};
  const DataLayout layout{DataLayout::NCHW};
  DenseTensorMeta meta(dtype, dims, layout);

  auto alloc =
      std::make_shared<experimental::DefaultAllocator>(platform::CPUPlace());

  DenseTensor dense_tensor(alloc, meta);
  float* data = dense_tensor.mutable_data<float>();
  data[0] = 1.0f;
  data[1] = 2.1f;

  framework::Tensor tensor;
  experimental::MovesStorage(&dense_tensor, &tensor);

  CHECK(dense_tensor.dtype() == pten::TransToPtenDataType(tensor.type()));
  CHECK(dense_tensor.layout() == pten::TransToPtenDataLayout(tensor.layout()));
  CHECK(platform::is_cpu_place(tensor.place()));

  CHECK(tensor.data<float>()[0] == 1.0f);
  CHECK(tensor.data<float>()[1] == 2.1f);

  auto dense_tensor_1 = experimental::MakePtenDenseTensor(tensor);
  CHECK(dense_tensor_1->dims() == dims);
  CHECK(dense_tensor_1->dtype() == dtype);
  CHECK(dense_tensor_1->layout() == layout);
  const float* data_1 = dense_tensor_1->data<float>();
  CHECK(data_1[0] == 1.0f);
  CHECK(data_1[1] == 2.1f);
}

TEST(PtenUtils, VarToPtTensor) {
  // 1. create Variable
  paddle::framework::Variable v;
  auto selected_rows = v.GetMutable<paddle::framework::SelectedRows>();
  paddle::framework::Tensor* value = selected_rows->mutable_value();
  auto* data = value->mutable_data<int>(paddle::framework::make_ddim({1, 1}),
                                        paddle::platform::CPUPlace());
  data[0] = 123;
  pten::Backend expect_backend = pten::Backend::CPU;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  expect_backend = pten::Backend::CUDA;
#endif
  auto tensor_def = pten::TensorArgDef(
      expect_backend, pten::DataLayout::NCHW, pten::DataType::INT32);
  // 2. test API
  auto tensor_x = experimental::MakePtenTensorBaseFromVar(v, tensor_def);
  // 3. check result
  ASSERT_EQ(tensor_x->dtype(), pten::DataType::INT32);
}

}  // namespace tests
}  // namespace paddle
