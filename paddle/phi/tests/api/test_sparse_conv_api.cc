/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/phi/api/include/api.h"

#include "paddle/phi/api/include/sparse_api.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/sparse_coo_tensor.h"

PD_DECLARE_KERNEL(sparse_conv3d, CPU, ALL_LAYOUT);

template <typename T>
void TestConv3dBase(const std::vector<int>& indices,
                    const std::vector<T>& features,
                    const phi::DDim& x_dims,
                    const std::vector<T>& kernel,
                    const phi::DDim& kernel_dims,
                    const std::vector<int>& correct_out_indices,
                    const std::vector<T>& correct_out_features,
                    const phi::DDim& correct_out_dims,
                    const int non_zero_num,
                    const std::vector<int>& paddings,
                    const std::vector<int>& strides,
                    const std::vector<int>& dilations,
                    const float diff = 1e-3) {
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  phi::DenseTensor indices_tensor(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT32, {4, non_zero_num}, phi::DataLayout::NCHW));
  memcpy(
      indices_tensor.data<int>(), indices.data(), indices.size() * sizeof(int));

  phi::DenseTensor features_tensor(
      alloc.get(),
      phi::DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                           {non_zero_num, in_channels},
                           phi::DataLayout::NHWC));
  memcpy(
      features_tensor.data<T>(), features.data(), features.size() * sizeof(T));

  auto x_tensor = std::make_shared<phi::SparseCooTensor>(
      indices_tensor, features_tensor, x_dims);
  paddle::experimental::Tensor x(x_tensor);

  auto kernel_tensor = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                           kernel_dims,
                           phi::DataLayout::NHWC));
  paddle::experimental::Tensor weight(kernel_tensor);

  memcpy(kernel_tensor->mutable_data<T>(paddle::platform::CPUPlace()),
         kernel.data(),
         kernel.size() * sizeof(T));

  if (!std::is_same<T, phi::dtype::float16>::value) {
    auto outs = paddle::experimental::sparse::conv3d(
        x, weight, paddings, dilations, strides, 1, false);

    auto out = std::dynamic_pointer_cast<phi::SparseCooTensor>(
        std::get<0>(outs).impl());
    ASSERT_EQ(correct_out_dims.size(), out->dims().size());
    for (int i = 0; i < correct_out_dims.size(); i++) {
      ASSERT_EQ(correct_out_dims[i], out->dims()[i]);
    }
    ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, out->nnz());

    int cmp_indices = memcmp(correct_out_indices.data(),
                             out->non_zero_indices().data<int>(),
                             correct_out_indices.size() * sizeof(int));
    ASSERT_EQ(cmp_indices, 0);

    for (uint64_t i = 0; i < correct_out_features.size(); i++) {
      float tmp = std::fabs(static_cast<float>(
          correct_out_features[i] - out->non_zero_elements().data<T>()[i]));
      ASSERT_LT(tmp, diff);
    }
  }
}

void TestConv3d(const std::vector<int>& indices,
                const std::vector<float>& features,
                const phi::DDim& x_dims,
                const std::vector<float>& kernel,
                const phi::DDim& kernel_dims,
                const std::vector<int>& correct_out_indices,
                const std::vector<float>& correct_out_features,
                const phi::DDim& correct_out_dims,
                const int non_zero_num,
                const std::vector<int>& paddings,
                const std::vector<int>& strides,
                const std::vector<int>& dilations) {
  // test float
  TestConv3dBase<float>(indices,
                        features,
                        x_dims,
                        kernel,
                        kernel_dims,
                        correct_out_indices,
                        correct_out_features,
                        correct_out_dims,
                        non_zero_num,
                        paddings,
                        strides,
                        dilations);
}

TEST(API, sparse_conv2d) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const int in_channels = 1;
  const int out_channels = 1;
  phi::DDim x_dims = {1, 1, 5, 5, in_channels};
  phi::DDim kernel_dims = {1, 3, 3, in_channels, out_channels};
  phi::DDim out_dims = {1, 1, 3, 3, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 3;
  std::vector<int> indices_flatten = {0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 2, 4};

  std::vector<float> features = {-0.79394531, -0.3125, -0.55029297};
  // 3*3*3=27
  std::vector<float> kernel = {0.65820312,
                               0.75048828,
                               0.21411133,
                               0.17370605,
                               0.85546875,
                               0.53076172,
                               0.28833008,
                               0.71044922,
                               0.00659943};

  std::vector<int> out_indices_flatten = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 2, 2, 2, 1, 2, 0, 1, 2};

  std::vector<float> out_features = {
      -0.17004, -0.71338, -0.00206, -0.22205, -0.09009};

  TestConv3d(indices_flatten,
             features,
             x_dims,
             kernel,
             kernel_dims,
             out_indices_flatten,
             out_features,
             out_dims,
             non_zero_num,
             paddings,
             strides,
             dilations);
}
