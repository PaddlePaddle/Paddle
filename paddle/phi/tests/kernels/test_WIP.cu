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
  paddle::platform::CUDAPlace gpu_place = paddle::platform::CUDAPlace(0);
  phi::GPUContext gpu_context(gpu_place);
  gpu_context.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(gpu_place, gpu_context.stream())
          .get());
  gpu_context.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  gpu_context.SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(gpu_place)
          .get());
  gpu_context.SetPinnedAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPinnedPlace())
          .get());
  gpu_context.PartialInitWithAllocator();
  phi::DataType dtype = phi::DataType::FLOAT32;
  int out_shape_1 = 1;

  phi::DenseTensor lse(dtype);
  phi::UniformKernel<float>(gpu_context, {2, 3, 4}, dtype, 0, 1, 1234, &lse);

  phi::DenseTensor out =
      phi::funcs::get_pad_lse<float>(gpu_context, &lse, out_shape_1, 32);
  EXPECT_EQ(out.dims()[2], 32);
  phi::CPUPlace cpu_place;
  phi::DenseTensor cpu_assert_tensor(dtype);
  cpu_assert_tensor.Resize({2, 3, 32});
  cpu_assert_tensor.mutable_data(cpu_place, dtype);
  paddle::framework::TensorCopySync(out, cpu_place, &cpu_assert_tensor);
  auto cpu_data = cpu_assert_tensor.data<float>();
  for (int i = 0; i < 2 * 3 * 32; ++i) {
    if (i % 32 >= 4) {
      EXPECT_TRUE(cpu_data[i] == std::numeric_limits<float>::infinity());
    }
  }

  phi::DenseTensor out_0 =
      phi::funcs::get_pad_lse<float>(gpu_context, &lse, out_shape_1, 32, true);
  EXPECT_EQ(out_0.dims()[2], 32);
  paddle::framework::TensorCopySync(out_0, cpu_place, &cpu_assert_tensor);
  cpu_data = cpu_assert_tensor.data<float>();
  for (int i = 0; i < 2 * 3 * 32; ++i) {
    if (i % 32 >= 1) {
      EXPECT_TRUE(cpu_data[i] == std::numeric_limits<float>::infinity());
    }
  }

  phi::DenseTensor out_1 =
      phi::funcs::get_pad_lse<float>(gpu_context, &lse, out_shape_1, 2, true);
  EXPECT_EQ(out_1.dims()[2], 4);
  paddle::framework::TensorCopySync(out_1, cpu_place, &cpu_assert_tensor);
  cpu_data = cpu_assert_tensor.data<float>();
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    std::cout << cpu_data[i] << std::endl;
    if (i % 4 >= out_shape_1) {
      EXPECT_TRUE(cpu_data[i] == std::numeric_limits<float>::infinity());
    }
  }
}
}  // namespace funcs
}  // namespace phi
