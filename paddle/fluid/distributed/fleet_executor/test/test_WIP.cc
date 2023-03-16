// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/fleet_executor/WIP.cu.h"
#include "paddle/phi/kernels/uniform_kernel.h"

namespace phi {

TEST(WIPTest, WIP) {
  phi::GPUPlace place;
  phi::GPUContext gpu_context(place);
  phi::DataType dtype = phi::DataType::FLOAT32;
  phi::DenseTensor lse(dtype);
  phi::UniformKernel<float>(place, {2, 3, 4}, dtype, 0, 1, 1234, &lse);
  EXPECT_TRUE(true == false);
}

}  // namespace phi
