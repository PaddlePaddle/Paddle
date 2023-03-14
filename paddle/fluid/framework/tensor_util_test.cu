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

#include "paddle/fluid/framework/tensor_util.h"
#include "gtest/gtest.h"
#include "paddle/fluid/operators/isfinite_op.h"
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
  buf[0] = INFINITY;
  buf[1] = 0.1;
  buf[2] = 0.2;
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

static __global__ void FillFinite(float* buf) {
  buf[0] = 0.0;
  buf[1] = 0.1;
  buf[2] = 0.2;
}

static __global__ void FillFinite(platform::float16* buf) {
  buf[0] = 0.0;
  buf[1] = 0.1;
  buf[2] = 0.2;
}

TEST(TensorContainsNAN, GPU) {
  paddle::platform::CUDAPlace gpu(0);
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  {
    phi::DenseTensor tensor;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillNAN, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsNAN(tensor));
  }
  {
    phi::DenseTensor tensor;
    paddle::platform::float16* buf =
        tensor.mutable_data<paddle::platform::float16>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillNAN, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsNAN(tensor));
  }
}

TEST(TensorContainsInf, GPU) {
  paddle::platform::CUDAPlace gpu(0);
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  {
    phi::DenseTensor tensor;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillInf, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsInf(tensor));
  }
  {
    phi::DenseTensor tensor;
    paddle::platform::float16* buf =
        tensor.mutable_data<paddle::platform::float16>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillInf, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    ASSERT_TRUE(TensorContainsInf(tensor));
  }
}

TEST(TensorIsfinite, GPU) {
  paddle::platform::CUDAPlace gpu(0);
  using paddle::platform::float16;
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  // contains inf
  {
    phi::DenseTensor tensor;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillInf, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    EXPECT_TRUE(!TensorIsfinite(tensor));
  }
  {
    phi::DenseTensor tensor;
    float16* buf = tensor.mutable_data<float16>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillInf, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    EXPECT_TRUE(!TensorIsfinite(tensor));
  }

  // contains nan
  {
    phi::DenseTensor tensor;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillNAN, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    EXPECT_TRUE(!TensorIsfinite(tensor));
  }
  {
    phi::DenseTensor tensor;
    float16* buf = tensor.mutable_data<float16>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillNAN, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    EXPECT_TRUE(!TensorIsfinite(tensor));
  }

  // all element are finite
  {
    phi::DenseTensor tensor;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(
        FillFinite, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillFinite<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    EXPECT_TRUE(TensorIsfinite(tensor));
  }
  {
    phi::DenseTensor tensor;
    float16* buf = tensor.mutable_data<float16>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(
        FillFinite, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillFinite<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    EXPECT_TRUE(TensorIsfinite(tensor));
  }
}

TEST(TensorContainsInf, GPUWithoutWait) {
  paddle::platform::CUDAPlace gpu(0);
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  {
    phi::DenseTensor tensor, out;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillInf, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    TensorContainsInf(tensor, &out);
    platform::CPUPlace cpu;
    phi::DenseTensor tmp;
    TensorCopy(out, cpu, *cuda_ctx, &tmp);
    cuda_ctx->Wait();
    ASSERT_EQ(tmp.data<bool>()[0], true);
  }
  {
    phi::DenseTensor tensor, out;
    paddle::platform::float16* buf =
        tensor.mutable_data<paddle::platform::float16>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillInf, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    TensorContainsInf(tensor, &out);
    platform::CPUPlace cpu;
    phi::DenseTensor tmp;
    TensorCopy(out, cpu, *cuda_ctx, &tmp);
    cuda_ctx->Wait();
    ASSERT_EQ(tmp.data<bool>()[0], true);
  }
}

TEST(TensorContainsNAN, GPUWithoutWait) {
  paddle::platform::CUDAPlace gpu(0);
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  {
    phi::DenseTensor tensor, out;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillNAN, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    TensorContainsNAN(tensor, &out);
    platform::CPUPlace cpu;
    phi::DenseTensor tmp;
    TensorCopy(out, cpu, *cuda_ctx, &tmp);
    cuda_ctx->Wait();
    ASSERT_EQ(tmp.data<bool>()[0], true);
  }
  {
    phi::DenseTensor tensor, out;
    paddle::platform::float16* buf =
        tensor.mutable_data<paddle::platform::float16>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillNAN, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    TensorContainsNAN(tensor, &out);
    platform::CPUPlace cpu;
    phi::DenseTensor tmp;
    TensorCopy(out, cpu, *cuda_ctx, &tmp);
    cuda_ctx->Wait();
    ASSERT_EQ(tmp.data<bool>()[0], true);
  }
}

TEST(TensorIsfinite, GPUWithoutWait) {
  paddle::platform::CUDAPlace gpu(0);
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* cuda_ctx = pool.GetByPlace(gpu);
  {
    phi::DenseTensor tensor, out;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillInf, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillInf<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    TensorIsfinite(tensor, &out);
    platform::CPUPlace cpu;
    phi::DenseTensor tmp;
    TensorCopy(out, cpu, *cuda_ctx, &tmp);
    cuda_ctx->Wait();
    EXPECT_EQ(tmp.data<bool>()[0], false);
  }
  {
    phi::DenseTensor tensor, out;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(FillNAN, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillNAN<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    TensorIsfinite(tensor, &out);
    platform::CPUPlace cpu;
    phi::DenseTensor tmp;
    TensorCopy(out, cpu, *cuda_ctx, &tmp);
    cuda_ctx->Wait();
    EXPECT_EQ(tmp.data<bool>()[0], false);
  }
  {
    phi::DenseTensor tensor, out;
    float* buf = tensor.mutable_data<float>({3}, gpu);
#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(
        FillFinite, dim3(1), dim3(1), 0, cuda_ctx->stream(), buf);
#else
    FillFinite<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
#endif
    cuda_ctx->Wait();
    TensorIsfinite(tensor, &out);
    platform::CPUPlace cpu;
    phi::DenseTensor tmp;
    TensorCopy(out, cpu, *cuda_ctx, &tmp);
    cuda_ctx->Wait();
    EXPECT_EQ(tmp.data<bool>()[0], true);
  }
}

}  // namespace framework
}  // namespace paddle
