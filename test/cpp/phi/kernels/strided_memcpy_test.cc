/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/strided_memcpy.h"
#include <array>

#include "gtest/gtest.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
namespace phi {
namespace tests {

TEST(StridedUtils, CheckedMulInt64Boundaries) {
  int64_t result = 0;
  EXPECT_FALSE(
      CheckedMulInt64(std::numeric_limits<int64_t>::lowest(), -1, &result));
  EXPECT_FALSE(
      CheckedMulInt64(-1, std::numeric_limits<int64_t>::lowest(), &result));
  EXPECT_TRUE(
      CheckedMulInt64(std::numeric_limits<int64_t>::lowest(), 1, &result));
  EXPECT_EQ(result, std::numeric_limits<int64_t>::lowest());
  EXPECT_FALSE(
      CheckedMulInt64(std::numeric_limits<int64_t>::lowest(), 2, &result));
  EXPECT_FALSE(
      CheckedMulInt64(std::numeric_limits<int64_t>::lowest(), -2, &result));
  EXPECT_FALSE(
      CheckedMulInt64(std::numeric_limits<int64_t>::max(), 2, &result));
  EXPECT_TRUE(CheckedMulInt64(-7, -9, &result));
  EXPECT_EQ(result, 63);
}

TEST(StridedUtils, HostAccumulateStridedView) {
  const std::vector<float> values = {1, 2, 3, 4, 5, 6};

  std::vector<float> non_overlapping(6, 0);
  HostAccumulateStridedView(
      values.data(), {2, 3}, {3, 1}, 0, non_overlapping.data());
  EXPECT_EQ(non_overlapping, values);

  std::vector<float> overlapping(4, 0);
  HostAccumulateStridedView(
      values.data(), {2, 3}, {1, 1}, 0, overlapping.data());
  EXPECT_EQ(overlapping, (std::vector<float>{1, 6, 8, 6}));

  std::vector<float> reversed(4, 0);
  HostAccumulateStridedView(values.data(), {4}, {-1}, 3, reversed.data());
  EXPECT_EQ(reversed, (std::vector<float>{4, 3, 2, 1}));

  std::vector<float> empty_storage = {7, 8};
  HostAccumulateStridedView(
      values.data(), {0, 3}, {1, 1}, 0, empty_storage.data());
  EXPECT_EQ(empty_storage, (std::vector<float>{7, 8}));
}

// Empty dims make ComputeStridedViewRange return early, which is the same
// guard StridedTensorAccumulateThroughStorage uses. Do not instantiate that
// helper here: it pulls in FillKernel/ContiguousKernel templates that are
// not exported from phi.dll, so Windows cannot link strided_memcpy_test.exe.
TEST(StridedUtils, ComputeStridedViewRangeEmptyShape) {
  auto empty_1d = ComputeStridedViewRange({0}, {1}, 0);
  EXPECT_TRUE(empty_1d.empty);

  auto empty_middle = ComputeStridedViewRange({2, 0, 3}, {3, 1, 1}, 0);
  EXPECT_TRUE(empty_middle.empty);

  auto nonempty = ComputeStridedViewRange({2, 3}, {3, 1}, 0);
  EXPECT_FALSE(nonempty.empty);
  EXPECT_EQ(nonempty.min_index, 0);
  EXPECT_EQ(nonempty.max_index, 5);
}

TEST(StridedMemcpy, CPUCrop) {
  // clang-format off
  int src[] = {// NOLINT
      0, 1, 2, 0, 0,
      0, 3, 4, 0, 0,
      0, 0, 0, 0, 0,
  };
  // clang-format on

  phi::DDim src_stride({5, 1});

  std::array<int, 4> dst = {};
  phi::DDim dst_dim({2, 2});
  phi::DDim dst_stride({2, 1});

  phi::CPUContext ctx;
  phi::funcs::StridedMemcpy<int>(
      ctx, src + 1, src_stride, dst_dim, dst_stride, dst.data());

  ASSERT_EQ(1, dst[0]);
  ASSERT_EQ(2, dst[1]);
  ASSERT_EQ(3, dst[2]);
  ASSERT_EQ(4, dst[3]);
}

TEST(StridedMemcpy, CPUConcat) {
  // clang-format off
  int src[] = { // NOLINT
      1, 2,
      3, 4
  };
  // clang-format on

  std::array<int, 8> dst = {};
  phi::DDim src_stride({2, 1});
  phi::DDim dst_dim({2, 2});
  phi::DDim dst_stride({4, 1});
  phi::CPUContext ctx;

  phi::funcs::StridedMemcpy<int>(
      ctx, src, src_stride, dst_dim, dst_stride, dst.data());
  phi::funcs::StridedMemcpy<int>(
      ctx, src, src_stride, dst_dim, dst_stride, dst.data() + 2);

  // clang-format off
  int expect_dst[] = { // NOLINT
      1, 2, 1, 2,
      3, 4, 3, 4
  };
  // clang-format on
  for (size_t i = 0; i < sizeof(expect_dst) / sizeof(int); ++i) {
    ASSERT_EQ(expect_dst[i], dst[i]);
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(StridedMemcpy, GPUCrop) {
  // clang-format off
  std::array<int, 15> src = {
      0, 1, 2, 0, 0,
      0, 3, 4, 0, 0,
      0, 0, 0, 0, 0,
  };
  // clang-format on

  phi::GPUPlace gpu0(0);
  phi::CPUPlace cpu;

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  auto src_allocation = phi::memory_utils::Alloc(gpu0, sizeof(src));

  int* gpu_src = reinterpret_cast<int*>(src_allocation->ptr());
  memory_utils::Copy(
      gpu0, gpu_src, cpu, src.data(), sizeof(src), ctx->stream());

  phi::DDim src_stride({5, 1});

  std::array<int, 4> dst = {};
  auto dst_allocation = phi::memory_utils::Alloc(gpu0, sizeof(dst));
  int* gpu_dst = reinterpret_cast<int*>(dst_allocation->ptr());

  phi::DDim dst_dim({2, 2});
  phi::DDim dst_stride({2, 1});

  phi::funcs::StridedMemcpy<int>(
      *ctx, gpu_src + 1, src_stride, dst_dim, dst_stride, gpu_dst);

  memory_utils::Copy(
      cpu, dst.data(), gpu0, gpu_dst, sizeof(dst), ctx->stream());
  ctx->Wait();

  ASSERT_EQ(1, dst[0]);
  ASSERT_EQ(2, dst[1]);
  ASSERT_EQ(3, dst[2]);
  ASSERT_EQ(4, dst[3]);
}

TEST(StridedMemcpy, GPUConcat) {
  // clang-format off
  std::array<int, 4> src = {
      1, 2,
      3, 4
  };
  // clang-format on

  phi::GPUPlace gpu0(0);
  phi::CPUPlace cpu;

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  auto gpu_src_allocation = phi::memory_utils::Alloc(gpu0, sizeof(src));
  int* gpu_src = reinterpret_cast<int*>(gpu_src_allocation->ptr());
  memory_utils::Copy(
      gpu0, gpu_src, cpu, src.data(), sizeof(src), ctx->stream());

  std::array<int, 8> dst = {};
  auto gpu_dst_allocation = phi::memory_utils::Alloc(gpu0, sizeof(dst));
  int* gpu_dst = reinterpret_cast<int*>(gpu_dst_allocation->ptr());

  phi::DDim src_stride({2, 1});
  phi::DDim dst_dim({2, 2});
  phi::DDim dst_stride({4, 1});

  phi::funcs::StridedMemcpy<int>(
      *ctx, gpu_src, src_stride, dst_dim, dst_stride, gpu_dst);
  phi::funcs::StridedMemcpy<int>(
      *ctx, gpu_src, src_stride, dst_dim, dst_stride, gpu_dst + 2);

  memory_utils::Copy(
      cpu, dst.data(), gpu0, gpu_dst, sizeof(dst), ctx->stream());
  ctx->Wait();

  // clang-format off
  std::array<int, 8> expect_dst = {
      1, 2, 1, 2,
      3, 4, 3, 4
  };
  // clang-format on
  for (size_t i = 0; i < sizeof(expect_dst) / sizeof(int); ++i) {
    ASSERT_EQ(expect_dst[i], dst[i]);
  }
}

#endif
}  // namespace tests
}  // namespace phi
