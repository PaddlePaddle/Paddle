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

#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

TEST(ProcessMesh, Ctor) {
  std::vector<int64_t> shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  int64_t size = shape[0] * shape[1];
  ProcessMesh process_mesh(shape, process_ids, dim_names);
  EXPECT_EQ(process_mesh.shape(), shape);
  EXPECT_EQ(process_mesh.process_ids(), process_ids);
  EXPECT_EQ(process_mesh.dim_names()[0], "x");
  EXPECT_EQ(process_mesh.dim_names()[1], "y");
  EXPECT_EQ(process_mesh.size(), size);
  EXPECT_EQ(process_mesh.ndim(), static_cast<int64_t>(shape.size()));
  EXPECT_EQ(process_mesh.dim_size(0), shape[0]);
  EXPECT_EQ(process_mesh.dim_size(-1), shape[1]);
  EXPECT_EQ(process_mesh.dim_size("x"), shape[0]);
  EXPECT_EQ(process_mesh.dim_size("y"), shape[1]);
  EXPECT_EQ(process_mesh.empty(), false);
  EXPECT_EQ(process_mesh.contains(0), true);
  EXPECT_EQ(process_mesh.contains(6), false);
  std::stringstream sstream;
  sstream << process_mesh;
  EXPECT_EQ(sstream.str(), process_mesh.to_string());
  auto proto = phi::distributed::to_proto(process_mesh);
  ProcessMesh new_process_mesh = ProcessMesh::from_proto(proto);
  EXPECT_EQ(process_mesh, new_process_mesh);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
