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

TEST(CopyTensor, float32) {
  typedef float DTYPE;
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1.0);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2.0);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::utils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3.0);
  tensor_dst->CopyToCpu<DTYPE>((float *)data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], 1.0, 1e-5);
  }
}

TEST(CopyTensor, int32) {
  typedef int32_t DTYPE;
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::utils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3);
  tensor_dst->CopyToCpu<DTYPE>((DTYPE *)data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], 1, 1e-5);
  }
}

TEST(CopyTensor, int64) {
  typedef int64_t DTYPE;
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::utils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3);
  tensor_dst->CopyToCpu<DTYPE>((DTYPE *)data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], 1, 1e-5);
  }
}

TEST(CopyTensor, int8) {
  typedef int8_t DTYPE;
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::utils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3);
  tensor_dst->CopyToCpu<DTYPE>((DTYPE *)data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], 1, 1e-5);
  }
}

TEST(CopyTensor, uint8) {
  typedef uint8_t DTYPE;
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::contrib::utils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::utils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3);
  tensor_dst->CopyToCpu<DTYPE>((DTYPE *)data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], 1, 1e-5);
  }
}

/*
TEST(CopyTensor, float16) {
  typedef paddle::platform::float16 DTYPE;
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::contrib::utils::CreateInferTensorForTest(
                        "tensor_src", PlaceType::kCPU,
                        static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::contrib::utils::CreateInferTensorForTest(
                        "tensor_dst", PlaceType::kCPU,
                        static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::utils::CopyTensor(*tensor_dst, *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3);
  tensor_dst->CopyToCpu<DTYPE>((DTYPE*)data_check.data());

  for (int i = 0; i < 6; i++) {
      EXPECT_NEAR(data_check[i], 1, 1e-5);
  }
}
*/

}  // namespace paddle_infer
