/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/cudnn_helper.h"
#include <gtest/gtest.h>

TEST(CudnnHelper, ScopedTensorDescriptor) {
  using paddle::platform::ScopedTensorDescriptor;
  using paddle::platform::DataLayout;

  ScopedTensorDescriptor tensor_desc;
  std::vector<int> shape = {2, 4, 6, 6};
  auto desc = tensor_desc.descriptor<float>(DataLayout::kNCHW, shape);

  miopenDataType_t type;
  std::vector<int> dims(4);
  std::vector<int> strides(4);
  paddle::platform::dynload::miopenGet4dTensorDescriptor(
      desc, &type, &dims[0], &dims[1], &dims[2], &dims[3],
      &strides[0], &strides[1], &strides[2], &strides[3]);

  for (size_t i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], shape[i]);
  }
  EXPECT_EQ(strides[3], 1);
  EXPECT_EQ(strides[2], 6);
  EXPECT_EQ(strides[1], 36);
  EXPECT_EQ(strides[0], 144);
}
TEST(CudnnHelper, ScopedFilterDescriptor) {
  using paddle::platform::ScopedFilterDescriptor;
  using paddle::platform::DataLayout;

  ScopedFilterDescriptor filter_desc;
  std::vector<int> shape = {2, 3, 3};

  miopenDataType_t type;
  std::vector<int> kernel(3);

  ScopedFilterDescriptor filter_desc_4d;
  std::vector<int> shape_4d = {2, 3, 3, 3};
  auto desc_4d = filter_desc.descriptor<float>(DataLayout::kNCDHW, shape_4d);

  std::vector<int> kernel_4d(4);
  std::vector<int> strides(4);
  paddle::platform::dynload::miopenGet4dTensorDescriptor(
      desc_4d, &type, &kernel_4d[0], &kernel_4d[1], &kernel_4d[2], &kernel_4d[3],
      &strides[0], &strides[1], &strides[2], &strides[3]);

  for (size_t i = 0; i < shape_4d.size(); ++i) {
    EXPECT_EQ(kernel_4d[i], shape_4d[i]);
  }
}

TEST(CudnnHelper, ScopedConvolutionDescriptor) {
  using paddle::platform::ScopedConvolutionDescriptor;

  ScopedConvolutionDescriptor conv_desc;
  std::vector<int> src_pads = {2, 2};
  std::vector<int> src_strides = {1, 1};
  std::vector<int> src_dilations = {1, 1};
  auto desc = conv_desc.descriptor<float>(src_pads, src_strides, src_dilations);

  miopenConvolutionMode_t mode;
  std::vector<int> pads(2);
  std::vector<int> strides(2);
  std::vector<int> dilations(2);
  paddle::platform::dynload::miopenGetConvolutionDescriptor(
      desc, &mode, &pads[0], &pads[1], &strides[0], &strides[1],
      &dilations[0], &dilations[1]);

  for (size_t i = 0; i < src_pads.size(); ++i) {
    EXPECT_EQ(pads[i], src_pads[i]);
    EXPECT_EQ(strides[i], src_strides[i]);
    EXPECT_EQ(dilations[i], src_dilations[i]);
  }
  EXPECT_EQ(mode, miopenConvolution);
}

TEST(CudnnHelper, ScopedPoolingDescriptor) {
  using paddle::platform::ScopedPoolingDescriptor;
  using paddle::platform::PoolingMode;

  ScopedPoolingDescriptor pool_desc;
  std::vector<int> src_kernel = {2, 2};
  std::vector<int> src_pads = {1, 1};
  std::vector<int> src_strides = {2, 2};
  auto desc = pool_desc.descriptor(PoolingMode::kMaximum, src_kernel, src_pads,
                                   src_strides);

  miopenPoolingMode_t mode;
  std::vector<int> kernel(2);
  std::vector<int> pads(2);
  std::vector<int> strides(2);
  paddle::platform::dynload::miopenGet2dPoolingDescriptor(
      desc, &mode, &kernel[0], &kernel[1], &pads[0], &pads[1],
      &strides[0], &strides[1]);

  for (size_t i = 0; i < src_pads.size(); ++i) {
    EXPECT_EQ(kernel[i], src_kernel[i]);
    EXPECT_EQ(pads[i], src_pads[i]);
    EXPECT_EQ(strides[i], src_strides[i]);
  }
  EXPECT_EQ(mode, miopenPoolingMax);
}
