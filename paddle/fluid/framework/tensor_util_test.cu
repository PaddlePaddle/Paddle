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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

static __global__ void FillNAN(float* buf) {
  buf[0] = 0.0;
  buf[1] = 0.1;
  buf[2] = NAN;
}

static __global__ void FillInf(float* buf) {
  buf[0] = 0.0;
  buf[1] = INFINITY;
  buf[2] = 0.5;
}

static __global__ void FillNAN(platform::float16* buf) {
  buf[0] = 0.0;
  buf[1] = 0.1;
  buf[2].x = 0x7fff;
}

static __global__ void FillInf(platform::float16* buf) {
  buf[0] = 0.0;
  buf[1].x = 0x7c00;
  buf[2] = 0.5;
}

TEST(TensorContainsNAN, GPU) {
  paddle::platform::CUDAPlace gpu(0);
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor tensor;
    float* buf = tensor.mutable_data<float>({3}, gpu);
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsNAN(tensor));
  }
  {
    Tensor tensor;
    paddle::platform::float16* buf =
        tensor.mutable_data<paddle::platform::float16>({3}, gpu);
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsNAN(tensor));
  }
}

TEST(TensorContainsInf, GPU) {
  paddle::platform::CUDAPlace gpu(0);
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor tensor;
    float* buf = tensor.mutable_data<float>({3}, gpu);
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsInf(tensor));
  }
  {
    Tensor tensor;
    paddle::platform::float16* buf =
        tensor.mutable_data<paddle::platform::float16>({3}, gpu);
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsInf(tensor));
  }
}

}  // namespace framework
}  // namespace paddle
