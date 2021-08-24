/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstring>
#include <numeric>
#include "gflags/gflags.h"
#include "paddle/fluid/inference/api/paddle_infer_contrib.h"
#include "paddle/fluid/inference/tests/api/trt_test_helper.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle_infer {

template <class DTYPE>
static void test_copy_tensor(PlaceType src_place, PlaceType dst_place) {
  paddle::framework::Scope scope;
  auto tensor_src =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_src", src_place, static_cast<void *>(&scope));
  auto tensor_dst =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_dst", dst_place, static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3);
  tensor_dst->CopyToCpu<DTYPE>(static_cast<DTYPE *>(data_check.data()));

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], 1, 1e-5);
  }
}

TEST(CopyTensor, float32) {
  test_copy_tensor<float>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<float>(PlaceType::kCPU, PlaceType::kGPU);
  test_copy_tensor<float>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, int32) {
  test_copy_tensor<int32_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<int32_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, int64) {
  test_copy_tensor<int64_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<int64_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, int8) {
  test_copy_tensor<int8_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<int8_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, uint8) {
  test_copy_tensor<uint8_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<uint8_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, float16) {
  paddle::framework::Scope scope;
  auto tensor_src =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));

  using paddle::platform::float16;
  std::vector<float16> data_src(6, float16(1.0));
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float16> data_dst(4, float16(2.0));
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<float16> data_check(6, float16(1.0));
  tensor_dst->CopyToCpu<float16>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == float16(1.0));
  }
}

TEST(CopyTensor, float16_gpu) {
  paddle::framework::Scope scope;
  auto tensor_src =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_src", PlaceType::kGPU, static_cast<void *>(&scope));
  auto tensor_dst =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  using paddle::platform::float16;
  std::vector<float16> data_src(6, float16(1.0));
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float16> data_dst(4, float16(2.0));
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<float16> data_check(6, float16(1.0));
  tensor_dst->CopyToCpu<float16>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == float16(1.0));
  }
}

TEST(CopyTensor, async_stream) {
  paddle::framework::Scope scope;
  auto tensor_src =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_src", PlaceType::kGPU, static_cast<void *>(&scope));
  auto tensor_dst =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  std::vector<float> data_src(6, 1.0);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float> data_dst(4, 2.0);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  cudaStream_t stream;
  paddle_infer::contrib::TensorUtils::CopyTensorAsync(
      *tensor_dst, *tensor_src, static_cast<void *>(&stream));

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  cudaStreamSynchronize(stream);

  std::vector<float> data_check(6, 1.0);
  tensor_dst->CopyToCpu<float>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], float(1.0), 1e-5);
  }
}

TEST(CopyTensor, async_callback) {
  paddle::framework::Scope scope;
  auto tensor_src =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst =
      paddle_infer::contrib::TensorUtils::CreateInferTensorForTest(
          "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  std::vector<float> data_src(6, 1.0);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float> data_dst(4, 2.0);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensorAsync(
      *tensor_dst, *tensor_src,
      [](void *cb_params) {
        Tensor *tensor = static_cast<Tensor *>(cb_params);
        EXPECT_EQ(tensor->shape().size(), (size_t)2);
        EXPECT_EQ(tensor->shape()[0], 2);
        EXPECT_EQ(tensor->shape()[1], 3);
      },
      static_cast<void *>(&(*tensor_dst)));

  cudaDeviceSynchronize();
}

}  // namespace paddle_infer
