/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

struct TransposeTestCase {
  // input
  std::vector<int64_t> input_shape;
  std::vector<std::vector<int64_t>> input_dims_mapping;

  // shape attribute
  std::vector<int> perm;

  // output
  std::vector<std::vector<int64_t>> expected_input_dims_mapping;
  std::vector<std::vector<int64_t>> expected_output_dims_mapping;
};

TEST(Transpose, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  //
  std::vector<TransposeTestCase> test_cases = {
      // input_shape, input_dims_mapping, perm,
      // expected_input_dims_mapping, expected_output_dims_mapping

      {{64, 48, 36, 24},
       {{0, 1}, {}, {}, {}},
       {1, 0, 2, 3},
       {{0, 1}, {}, {}, {}},
       {{}, {0, 1}, {}, {}}},
      {{64, 48, 36, 24},
       {{0, 1}, {}, {}, {}},
       {0, 1, 2, 3},
       {{0, 1}, {}, {}, {}},
       {{0, 1}, {}, {}, {}}},
      {{64, 48, 36, 24},
       {{}, {}, {0, 1}, {}},
       {0, 2, 3, 1},
       {{}, {}, {0, 1}, {}},
       {{}, {0, 1}, {}, {}}},
      {{64, 48, 36, 24},
       {{}, {}, {0, 1}, {}},
       {-1, 0, -2, 1},
       {{}, {}, {0, 1}, {}},
       {{}, {}, {0, 1}, {}}},
  };

  for (const auto& tc : test_cases) {
    TensorDistAttr t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(tc.input_dims_mapping);
    t_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.input_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.input_shape), t_dist_attr);

    // test forward
    phi::distributed::SpmdInfo forward_spmd_info =
        phi::distributed::TransposeInferSpmd(x, tc.perm);
    EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
    EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(forward_spmd_info.first[0],
                             tc.expected_input_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.second[0],
                             tc.expected_output_dims_mapping);
  }
}
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
