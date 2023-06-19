// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/buffer.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/backends/cuda_util.h"
#endif
#include <gtest/gtest.h>

#include <vector>

namespace cinn {
namespace hlir {
namespace framework {

TEST(Buffer, basic) {
  Buffer buffer(common::DefaultHostTarget());
  buffer.Resize(10 * sizeof(float));
  auto* data = reinterpret_cast<float*>(buffer.data()->memory);
  for (int i = 0; i < 10; i++) data[i] = i;
}

#ifdef CINN_WITH_CUDA
TEST(Buffer, nvgpu) {
  const int num_elements = 10;
  Buffer buffer(common::DefaultNVGPUTarget());
  buffer.Resize(num_elements * sizeof(float));
  auto* data = reinterpret_cast<float*>(buffer.data()->memory);
  std::vector<float> host_data(num_elements);
  std::vector<float> host_target(num_elements, 0);

  for (int i = 0; i < num_elements; i++) {
    host_data[i] = i;
  }
  LOG(INFO) << "Cuda copy data";
  CUDA_DRIVER_CALL(cuMemcpy(reinterpret_cast<CUdeviceptr>(data),
                            reinterpret_cast<CUdeviceptr>(host_data.data()),
                            num_elements * sizeof(float)));
  CUDA_DRIVER_CALL(cuMemcpy(reinterpret_cast<CUdeviceptr>(host_target.data()),
                            reinterpret_cast<CUdeviceptr>(data),
                            num_elements * sizeof(float)));
  for (int i = 0; i < num_elements; i++) {
    ASSERT_EQ(host_target[i], i);
  }
}
#endif

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
