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

#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(FusedLinearParamGradAddSPMDRule, Ctor) {
  // build input data class

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  // b s h
  std::vector<int64_t> x_shape = {2, 512, 1024};
  std::vector<int64_t> out_shape = {2, 512, 2048};
  std::vector<int64_t> weight_shape = {1024, 2048};
  std::vector<int64_t> bias_shape = {2048};

  // test mp col split
  {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1}));
    x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));

    TensorDistAttr out_dist_attr = TensorDistAttr();
    out_dist_attr.set_process_mesh(process_mesh);
    out_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, 1}));
    out_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));

    phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
    phi::distributed::DistMetaTensor out(phi::make_ddim(out_shape),
                                         out_dist_attr);
    phi::distributed::DistMetaTensor dweight;
    phi::distributed::DistMetaTensor dbias;
    for (int i = 0; i < 3; i++) {
      auto spmd_info =
          FusedLinearParamGradAddInferSpmd(x, out, dweight, dbias, 0, true);
      check_dim_mapping(spmd_info.second[0], {-1, 1});
      check_partial_dims(spmd_info.second[0], {0});
      check_dim_mapping(spmd_info.second[1], {1});
      check_partial_dims(spmd_info.second[1], {0});
      dweight = phi::distributed::DistMetaTensor(
          phi::make_ddim(weight_shape),
          PADDLE_GET_CONST(TensorDistAttr, spmd_info.second[0]));
      dbias = phi::distributed::DistMetaTensor(
          phi::make_ddim(bias_shape),
          PADDLE_GET_CONST(TensorDistAttr, spmd_info.second[1]));
    }
  }

  // test mp row split
  {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, 1}));
    x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));

    TensorDistAttr out_dist_attr = TensorDistAttr();
    out_dist_attr.set_process_mesh(process_mesh);
    out_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1}));
    out_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));

    phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
    phi::distributed::DistMetaTensor out(phi::make_ddim(out_shape),
                                         out_dist_attr);
    phi::distributed::DistMetaTensor dweight;
    phi::distributed::DistMetaTensor dbias;
    for (int i = 0; i < 3; i++) {
      auto spmd_info =
          FusedLinearParamGradAddInferSpmd(x, out, dweight, dbias, 0, true);
      check_dim_mapping(spmd_info.second[0], {1, -1});
      check_partial_dims(spmd_info.second[0], {0});
      check_dim_mapping(spmd_info.second[1], {-1});
      check_partial_dims(spmd_info.second[1], {0});
      dweight = phi::distributed::DistMetaTensor(
          phi::make_ddim(weight_shape),
          PADDLE_GET_CONST(TensorDistAttr, spmd_info.second[0]));
      dbias = phi::distributed::DistMetaTensor(
          phi::make_ddim(bias_shape),
          PADDLE_GET_CONST(TensorDistAttr, spmd_info.second[1]));
    }
  }
}
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
