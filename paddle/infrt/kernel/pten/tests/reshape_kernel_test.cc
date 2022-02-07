/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"

#include "paddle/infrt/backends/host/pten_allocator.h"
#include "paddle/infrt/backends/host/pten_context.h"
#include "paddle/pten/kernels/reshape_kernel.h"

namespace infrt {
namespace kernels {
namespace tests {

TEST(pten, reshape) {
  auto allocator = backends::HostPtenAllocator();
  auto context = backends::HostPtenContext();
  context.SetDeviceAllocator(&allocator);
  context.SetHostAllocator(&allocator);
  auto tensor_meta =
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 2, 2, 3}),
                            pten::DataLayout::NCHW);
  auto dense_x = pten::DenseTensor(&allocator, std::move(tensor_meta));
  auto* dense_x_data = static_cast<float*>(
      dense_x.AllocateFrom(&allocator, pten::DataType::FLOAT32));

  // The writing is cumbersome and needs to be adjusted.
  auto out = pten::Reshape<float, backends::HostPtenContext::Base>(
      context, dense_x, {12, 3});
  std::vector<int64_t> expect_shape = {12, 3};
  ASSERT_EQ(out.dims()[0], expect_shape[0]);
  ASSERT_EQ(out.dims()[1], expect_shape[1]);
  ASSERT_EQ(out.numel(), 36);
  ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  bool value_equal = true;
  auto* dense_out_data = out.data<float>();
  for (int i = 0; i < dense_x.numel(); i++) {
    if (std::abs(dense_x_data[i] - dense_out_data[i]) > 1e-6f)
      value_equal = false;
  }
  ASSERT_EQ(value_equal, true);
}

}  // namespace tests
}  // namespace kernels
}  // namespace infrt
