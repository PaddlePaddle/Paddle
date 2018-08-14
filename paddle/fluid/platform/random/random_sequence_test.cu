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
#include "paddle/fluid/platform/random/normal_distribution.h"
#include "paddle/fluid/platform/random/random_sequence.h"
#include "paddle/fluid/platform/random/uniform_distribution.h"
#include "thrust/device_vector.h"

namespace paddle {
namespace platform {
namespace random {

template <typename T>
struct FillValue {
  T* val_;
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

}  // namespace random
}  // namespace platform
}  // namespace paddle
