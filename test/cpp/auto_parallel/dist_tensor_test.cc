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

#include "gtest/gtest.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "test/cpp/phi/core/allocator.h"

namespace phi {
namespace distributed {
namespace auto_parallel {
namespace tests {

TEST(dist_tensor, constructor) {
  auto fancy_allocator =
      std::unique_ptr<Allocator>(new phi::tests::FancyAllocator);
  auto* alloc = fancy_allocator.get();

  DataType dtype{DataType::FLOAT16};
  DDim dims({3, 4});
  DenseTensorMeta meta(dtype, dims);

  auto dist_attr = std::make_shared<TensorDistAttr>(phi::vectorize(dims));

  DistTensor x1(alloc, meta, dist_attr);
  EXPECT_TRUE(x1.defined());
  EXPECT_TRUE(x1.initialized());

  DistTensor x2(alloc, DenseTensorMeta(dtype, dims), dist_attr);
  EXPECT_TRUE(x2.defined());
  EXPECT_TRUE(x2.initialized());

  DistTensor x3(x2.value().Holder(), meta, dist_attr);
  EXPECT_TRUE(x3.defined());
  EXPECT_TRUE(x3.initialized());

  auto a = std::make_shared<DenseTensor>(alloc, DenseTensorMeta(dtype, dims));
  DistTensor x4(a, dist_attr);
  EXPECT_TRUE(x4.defined());
  EXPECT_TRUE(x4.initialized());
}

}  // namespace tests
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
