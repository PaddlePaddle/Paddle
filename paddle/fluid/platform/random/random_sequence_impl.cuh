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

#pragma once
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/random/philox_engine.h"
namespace paddle {
namespace platform {
namespace random {

template <>
struct RandomSequence<CUDADeviceContext> {
  template <typename Callback, typename Distribution>
  struct ForRangeFunctor {
    HOSTDEVICE ForRangeFunctor(Callback callback, uint64_t seed, size_t length,
                               Distribution dist)
        : callback_(callback), seed_(seed), length_(length), dist_(dist) {}

    HOSTDEVICE inline void operator()(size_t i) {
      Philox32x4 engine(seed_);
      engine.Discard(i);
      auto dist = dist_;
      size_t offset = i * Distribution::N;
      if (offset + Distribution::N < length_) {
        #pragma unroll (Distribution::N)
        for (size_t j = 0; j < Distribution::N; ++j, ++offset) {
          callback_(offset, dist(engine));
        }
      } else {
        for (size_t j = 0; j < Distribution::N; ++j, ++offset) {
          if (offset >= length_) {
            break;
          }
          callback_(offset, dist(engine));
        }
      }
    }

   private:
    Callback callback_;
    uint64_t seed_;
    size_t length_;
    Distribution dist_;
  };

  template <typename Distribution, typename Callback>
  void operator()(const CUDADeviceContext& ctx, uint64_t seed, size_t length,
                  Distribution dist, Callback callback) {
    Philox32x4 engine(seed);
    size_t kern_size = length / Distribution::N;

    ForRange<CUDADeviceContext> for_range(ctx, kern_size);
    ForRangeFunctor<Callback, Distribution> functor(callback, seed, length,
                                                    dist);
    for_range(functor);
  }
};

}  // namespace random
}  // namespace platform
}  // namespace paddle
