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
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/funcs/WIP.cu.h"
#include "paddle/phi/kernels/uniform_kernel.h"

namespace phi {
namespace funcs {

TEST(WIPTest, WIP) {
  phi::GPUPlace gpu_place;
  phi::GPUContext gpu_context(gpu_place);
  phi::DataType dtype = phi::DataType::FLOAT32;
  phi::DenseTensor lse(dtype);
  phi::UniformKernel<float>(gpu_context, {2, 3, 4}, dtype, 0, 1, 1234, &lse);

  phi::CPUPlace cpu_place;
  phi::DenseTensor cpu_assert_tensor(dtype);
  cpu_assert_tensor.Resize({2, 3, 4});
  cpu_assert_tensor.mutable_data(cpu_place, dtype);
  paddle::framework::TensorCopySync(lse, cpu_place, &cpu_assert_tensor);
  auto cpu_data = cpu_assert_tensor.data<float>();
  for (int i = 0; i < 2 * 3 * 4; ++i) std::cout << cpu_data[i] << std::endl;
  EXPECT_TRUE(true == false);
}
}  // namespace funcs
}  // namespace phi
