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
#include <stdint.h>
#include <stdlib.h>
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {
namespace random {

template <typename DeviceContext>
struct RandomSequence {
  template <typename Distribution, typename Callback>
  void operator()(const DeviceContext& ctx, uint64_t seed, size_t length,
                  Distribution dist, Callback callback);
};

template <typename DeviceContext, typename Distribution, typename T>
inline void RandomFill(const DeviceContext& ctx, uint64_t seed,
                       Distribution dist, T* data, size_t length);

}  // namespace random
}  // namespace platform
}  // namespace paddle

#include "paddle/fluid/platform/random/random_sequence_impl.h"
#ifdef __CUDACC__
#include "paddle/fluid/platform/random/random_sequence_impl.cuh"
#endif
