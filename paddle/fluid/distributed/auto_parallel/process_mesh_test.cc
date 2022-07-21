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

#include "paddle/fluid/distributed/auto_parallel/process_mesh.h"
#include <iostream>
#include "gtest/gtest.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(ProcessMesh, Ctor) {
  std::vector<int64_t> shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  std::string device_type = "GPU";
  int64_t size = shape[0] * shape[1];
  ProcessMesh process_mesh(shape, process_ids, dim_names, device_type);
  EXPECT_EQ(process_mesh.shape(), shape);
  EXPECT_EQ(process_mesh.process_ids(), process_ids);
  EXPECT_EQ(process_mesh.dim_names()[0], "x");
  EXPECT_EQ(process_mesh.dim_names()[1], "y");
  EXPECT_EQ(process_mesh.device_type(), device_type);
  EXPECT_EQ(process_mesh.size(), size);
  EXPECT_EQ(process_mesh.ndim(), static_cast<int64_t>(shape.size()));
  EXPECT_EQ(process_mesh.dim_size(0), shape[0]);
  EXPECT_EQ(process_mesh.dim_size(-1), shape[1]);
  EXPECT_EQ(process_mesh.dim_size("x"), shape[0]);
  EXPECT_EQ(process_mesh.dim_size("y"), shape[1]);
  std::cout << process_mesh << std::endl;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
