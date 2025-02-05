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

  auto dist_attr = TensorDistAttr(common::vectorize(dims));

  std::vector<int64_t> mesh_shape = {1};
  std::vector<int64_t> process_ids = {0};
  std::vector<std::string> dim_names = {"x"};
  ProcessMesh mesh(mesh_shape, process_ids, dim_names);
  dist_attr.set_process_mesh(mesh);

  // copy construct
  std::shared_ptr<DenseTensor> x1 = std::make_shared<DenseTensor>(alloc, meta);
  DistTensor dist_x1(x1, dist_attr);
  EXPECT_TRUE(dist_x1.defined());
  EXPECT_TRUE(dist_x1.initialized());
  EXPECT_TRUE(dist_x1.valid());
  EXPECT_EQ(dist_x1.numel(), 12L);
  EXPECT_EQ(dist_x1.local_dims()[0], 3L);
  EXPECT_EQ(dist_x1.local_dims()[1], 4L);

  // empty construct
  DistTensor dist_x2(dims, dist_attr);
  EXPECT_TRUE(!dist_x2.defined());
  EXPECT_TRUE(!dist_x2.initialized());
  // allocate error test
  bool caught_exception = false;
  try {
    dist_x2.AllocateFrom(alloc, phi::DataType::FLOAT32, 12L, false);
  } catch (common::enforce::EnforceNotMet& error) {
    caught_exception = true;
    EXPECT_NE(std::string(error.what()).find("Unavailable"), 0UL);
  }
  EXPECT_TRUE(caught_exception);
}

}  // namespace tests
}  // namespace distributed
}  // namespace phi
