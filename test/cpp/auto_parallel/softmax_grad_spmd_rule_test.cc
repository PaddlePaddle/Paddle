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

TEST(SoftmaxGradInferSpmd, Ctor) {
  // Sharding along axes besides softmax axis.
  std::vector<int64_t> x_shape = {32, 48};
  std::vector<int64_t> out_grad_shape = {32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out_grad_dist_attr = TensorDistAttr();
  out_grad_dist_attr.set_process_mesh(process_mesh);
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
  out_grad_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor out_grad(phi::make_ddim(x_shape),
                                            out_grad_dist_attr);
  int axis = 1;

  auto spmdinfo = SoftmaxGradInferSpmd(x, out_grad, axis);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]), std::vector<int64_t>({1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]), std::vector<int64_t>({1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({1, -1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);
  VLOG(4) << "Test SoftmaxGradInferSpmd sharding on other axes." << std::endl
          << std::endl
          << std::endl;

  // Sharding along softmax axis.
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1}));
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1}));
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape),
                                              out_grad_dist_attr);
  axis = 1;

  spmdinfo = SoftmaxGradInferSpmd(x, out_grad, axis);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({-1, -1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);
  VLOG(4) << "Test SoftmaxGradInferSpmd sharding on softmax axis." << std::endl
          << std::endl
          << std::endl;

  // Sharding on multi axes.
  x_shape = {10, 32, 48, 24};
  out_grad_shape = {10, 32, 48, 24};
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1, -1}));
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1, -1}));
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape),
                                              out_grad_dist_attr);
  axis = 1;

  spmdinfo = SoftmaxGradInferSpmd(x, out_grad, axis);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({0, -1, -1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({0, -1, -1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({0, -1, -1, -1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);
  VLOG(4) << "Test SoftmaxGradInferSpmd sharding on multi axes." << std::endl
          << std::endl
          << std::endl;

  // Sharding on multi axes.
  x_shape = {10, 32, 48, 24};
  out_grad_shape = {10, 32, 48, 24};
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  out_grad_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 1, -1}));
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  out_grad = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape),
                                              out_grad_dist_attr);
  axis = 1;

  spmdinfo = SoftmaxGradInferSpmd(x, out_grad, axis);

  EXPECT_EQ(spmdinfo.first.size(), 2UL);
  EXPECT_EQ(spmdinfo.second.size(), 1UL);

  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({0, -1, 1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[1]),
            std::vector<int64_t>({0, -1, 1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({0, -1, 1, -1}));
  EXPECT_DOUBLE_EQ(
      PADDLE_GET_CONST(TensorDistAttr, spmdinfo.second[0]).is_partial(), false);
  VLOG(4) << "Test SoftmaxGradInferSpmd sharding on multi axes." << std::endl
          << std::endl
          << std::endl;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
