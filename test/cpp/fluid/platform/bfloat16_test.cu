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

#include "paddle/phi/common/bfloat16.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>

#include "paddle/fluid/framework/lod_tensor.h"

#if defined(PADDLE_CUDA_BF16)
namespace paddle {
namespace platform {

using bfloat16 = phi::dtype::bfloat16;
using namespace phi::dtype;  // NOLINT

TEST(bfloat16, convert_float32_to_bfloat16_on_gpu) {
  // Convert float32 to bfloat16
  EXPECT_EQ((bfloat16(1.0f)).x, 0x3f80);
  EXPECT_EQ((bfloat16(0.5f)).x, 0x3f00);
  EXPECT_EQ((bfloat16(0.33333f)).x, 0x3eab);
  EXPECT_EQ((bfloat16(0.0f)).x, 0x0000);
  EXPECT_EQ((bfloat16(-0.0f)).x, 0x8000);
  EXPECT_EQ((bfloat16(65536.0f)).x, 0x4780);
}

TEST(bfloat16, assignment_operator_on_gpu) {
  // Assignment operator
  bfloat16 v_assign;
  v_assign = bfloat16(1.0f).to_nv_bfloat16();
  EXPECT_EQ(v_assign.x, 0x3f80);
  v_assign = 0.33333;
  EXPECT_EQ(v_assign.x, 0x3eab);
}

TEST(bfloat16, convert_bfloat16_to_float32_on_gpu) {
  // Conversion operator
  EXPECT_EQ(static_cast<float>(bfloat16(0.5f)), 0.5f);
  EXPECT_NEAR(static_cast<double>(bfloat16(0.33333)), 0.33333, 0.01);
  EXPECT_EQ(static_cast<int>(bfloat16(-1)), -1);
  EXPECT_EQ(static_cast<bool>(bfloat16(true)), true);
}

TEST(bfloat16, dense_tensor_on_gpu) {
  phi::DenseTensor src_tensor;
  phi::DenseTensor gpu_tensor;
  phi::DenseTensor dst_tensor;

  bfloat16 *src_ptr =
      src_tensor.mutable_data<bfloat16>(common::make_ddim({2, 2}), CPUPlace());

  bfloat16 arr[4] = {
      bfloat16(1.0f), bfloat16(0.5f), bfloat16(0.33333f), bfloat16(0.0f)};
  memcpy(src_ptr, arr, 4 * sizeof(bfloat16));

  // CPU DenseTensor to GPU DenseTensor
  phi::GPUPlace gpu_place(0);
  phi::GPUContext gpu_ctx(gpu_place);
  gpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, gpu_ctx.stream())
                           .get());
  gpu_ctx.PartialInitWithAllocator();
  framework::TensorCopy(src_tensor, gpu_place, gpu_ctx, &gpu_tensor);

  // GPU DenseTensor to CPU DenseTensor
  framework::TensorCopy(gpu_tensor, CPUPlace(), gpu_ctx, &dst_tensor);

  // Sync before comparing DenseTensors
  gpu_ctx.Wait();
  const bfloat16 *dst_ptr = dst_tensor.data<bfloat16>();
  ASSERT_NE(src_ptr, dst_ptr);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(src_ptr[i].x, dst_ptr[i].x);
  }
}

TEST(bfloat16, isinf) {
  bfloat16 a;
  a.x = 0x7f80;
  bfloat16 b = bfloat16(INFINITY);
  bfloat16 c = static_cast<bfloat16>(INFINITY);
  EXPECT_EQ(std::isinf(a), true);
  EXPECT_EQ(std::isinf(b), true);
  EXPECT_EQ(std::isinf(c), true);
}

TEST(bfloat16, isnan) {
  bfloat16 a;
  a.x = 0x7fff;
  bfloat16 b = bfloat16(NAN);
  bfloat16 c = static_cast<bfloat16>(NAN);
  EXPECT_EQ(std::isnan(a), true);
  EXPECT_EQ(std::isnan(b), true);
  EXPECT_EQ(std::isnan(c), true);
}

TEST(bfloat16, cast) {
  bfloat16 a;
  a.x = 0x0070;
  auto b = a;
  {
    // change semantic, keep the same value
    bfloat16 c = reinterpret_cast<bfloat16 &>(reinterpret_cast<unsigned &>(b));
    EXPECT_EQ(b, c);
  }

  {
    // use uint32 low 16 bit store float16
    uint32_t c = reinterpret_cast<uint32_t &>(b);
    bfloat16 d;
    d.x = c;
    EXPECT_EQ(b, d);
  }
}

}  // namespace platform
}  // namespace paddle
#endif
