// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/platform/random/identity_distribution.h"
#include "paddle/fluid/platform/random/normal_distribution.h"
#include "paddle/fluid/platform/random/random_sequence.h"
#include "paddle/fluid/platform/random/uniform_distribution.h"
#include "thrust/device_vector.h"

namespace paddle {
namespace platform {
namespace random {

template <typename T>
struct FillValue {
  T *val_;
  HOSTDEVICE void operator()(size_t i, T val) { val_[i] = val; }
};

template <typename Dist>
void TestMain(Dist dist) {
  using T = typename Dist::ResultType;
  constexpr size_t length = 65536;
  std::vector<T> cpu(length);

  {
    RandomSequence<CPUDeviceContext> rand_seq;
    FillValue<T> fill_value;
    fill_value.val_ = cpu.data();
    auto tmp = dist;
    rand_seq(CPUDeviceContext(), 0, length, dist, fill_value);
  }

  thrust::device_vector<T> gpu(length);
  {
    RandomSequence<CUDADeviceContext> rand_seq;
    FillValue<T> fill_value;
    fill_value.val_ = gpu.data().get();
    CUDADeviceContext ctx(CUDAPlace(0));
    rand_seq(ctx, 0, length, dist, fill_value);
    ctx.Wait();
  }
  for (size_t i = 0; i < length; ++i) {
    ASSERT_NEAR(cpu[i], gpu[i], 1e-5);
  }
}

TEST(RandomSequence, UniformSame) {
  TestMain(UniformRealDistribution<float>(-1, 1));
  TestMain(UniformRealDistribution<double>(-1, 1));
}

TEST(RandomSequence, NormalSame) {
  TestMain(NormalDistribution<float>(0, 1));
  TestMain(NormalDistribution<double>(0, 1));
}

TEST(RandomSequence, IdentityCPUGPUSame) {
  TestMain(IdentityDistribution<uint32_t>());
  TestMain(IdentityDistribution<uint16_t>());
}

template <typename Dist>
void TestIdentityMean(Dist dist) {
  using T = typename Dist::ResultType;
  constexpr size_t length = 1U << 18;
  std::vector<T> cpu(length);
  RandomFill(CPUDeviceContext(), 0, dist, cpu.data(), length);

  __uint128_t sum = 0;
  for (auto &item : cpu) {
    sum += item;
  }
  sum /= length;

  ASSERT_LE(
      std::abs(static_cast<double>(sum) - static_cast<double>(Dist::Max / 2)) /
          (Dist::Max / 2),
      1e-3);
}

TEST(RandomSequence, IdentityMean) {
  TestIdentityMean(IdentityDistribution<uint32_t>());
}

TEST(RandomSequenceIdentity, IdentitySame) {
  constexpr size_t length = 1U << 18;
  std::vector<uint32_t> cpu(length);
  RandomFill(CPUDeviceContext(), 0, IdentityDistribution<uint32_t>(),
             cpu.data(), length);

  std::vector<uint32_t> cpu2;
  cpu2.reserve(length);
  Philox32x4 engine(0);
  for (size_t i = 0; i < (length << 2); ++i) {
    auto result = engine();
    for (size_t j = 0; j < result.size; ++j) {
      cpu2.emplace_back(result[j]);
    }
  }

  for (size_t i = 0; i < length; ++i) {
    if (cpu[i] != cpu2[i]) {
      std::cerr << i << std::endl;
    }
    ASSERT_EQ(cpu[i], cpu2[i]);
  }
}

}  // namespace random
}  // namespace platform
}  // namespace paddle
