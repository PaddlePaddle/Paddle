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

#pragma once

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace operators {

inline void GetSeedDataAndIncrement(const phi::GPUContext& dev_ctx,
                                    const phi::DenseTensor* seed,
                                    const bool is_fix_seed,
                                    const int seed_val,
                                    const int offset,
                                    uint64_t* seed_data,
                                    uint64_t* increment) {
  int device_id = dev_ctx.GetPlace().GetDeviceId();
  auto gen_cuda = framework::DefaultCUDAGenerator(device_id);

  if (seed) {
    phi::DenseTensor seed_cpu_tensor;
    paddle::framework::TensorCopySync(
        *seed, platform::CPUPlace(), &seed_cpu_tensor);
    *seed_data = static_cast<uint64_t>(seed_cpu_tensor.data<int>()[0]);
    *increment = offset;
  } else if (!is_fix_seed) {
    auto seed_offset = gen_cuda->IncrementOffset(offset);
    *seed_data = seed_offset.first;
    *increment = seed_offset.second;
  } else {
    *seed_data = seed_val;
    *increment = offset;
  }
}

}  // namespace operators
}  // namespace paddle
