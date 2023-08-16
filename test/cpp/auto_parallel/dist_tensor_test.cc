/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

#include <iostream>

#include "gtest/gtest.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "test/cpp/phi/core/allocator.h"

namespace phi {
namespace distributed {
namespace tests {

TEST(dist_tensor, constructor) {
  auto fancy_allocator =
      std::unique_ptr<Allocator>(new phi::tests::FancyAllocator);
  auto* alloc = fancy_allocator.get();

  DataType dtype{DataType::FLOAT16};
  DDim dims({3, 4});
  DenseTensorMeta meta(dtype, dims);

  auto dist_attr = TensorDistAttr(phi::vectorize(dims));

  // copy construct
  DenseTensor x1(alloc, meta);
  DistTensor dist_x1(dims, dist_attr, x1);
  EXPECT_TRUE(dist_x1.defined());
  EXPECT_TRUE(dist_x1.initialized());

  // empty construct
  DistTensor dist_x2(dims, dist_attr);
  EXPECT_TRUE(!dist_x2.defined());
  EXPECT_TRUE(!dist_x2.initialized());
}

}  // namespace tests
}  // namespace distributed
}  // namespace phi
