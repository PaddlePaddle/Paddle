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

TEST(CrossEntropyInferSpmd, Ctor) {
  std::vector<int64_t> x_shape = {32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr label_dist_attr = TensorDistAttr();
  label_dist_attr.set_process_mesh(process_mesh);
  label_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));
  label_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  // forward
  {
    phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
    phi::distributed::DistMetaTensor label(phi::make_ddim(x_shape),
                                           label_dist_attr);
    int axis = 1;

    auto spmdinfo =
        CrossEntropyWithSoftmaxInferSpmd(x, label, false, true, true, 1, axis);

    EXPECT_EQ(spmdinfo.first.size(), 2UL);
    EXPECT_EQ(spmdinfo.second.size(), 2UL);
    check_dim_mapping(spmdinfo.first[0], {0, -1});
    check_dim_mapping(spmdinfo.first[1], {0, -1});
    check_dim_mapping(spmdinfo.second[0], {0, -1});
    check_dim_mapping(spmdinfo.second[1], {0, -1});
    check_partial_dims(spmdinfo.second[0], {});

    VLOG(4) << "Test CrossEntropyWithSoftmaxInferSpmd sharding on other axes."
            << std::endl
            << std::endl
            << std::endl;
  }

  // test sharding along softmax axis.
  {
    x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1}));
    label_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));
    phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
    phi::distributed::DistMetaTensor label(phi::make_ddim(x_shape),
                                           label_dist_attr);
    int axis = 1;

    auto spmdinfo =
        CrossEntropyWithSoftmaxInferSpmd(x, label, false, true, true, 1, axis);

    EXPECT_EQ(spmdinfo.first.size(), 2UL);
    EXPECT_EQ(spmdinfo.second.size(), 2UL);
    check_dim_mapping(spmdinfo.first[0], {0, -1});
    check_dim_mapping(spmdinfo.first[1], {0, -1});
    check_dim_mapping(spmdinfo.second[0], {0, -1});
    check_dim_mapping(spmdinfo.second[1], {0, -1});
    check_partial_dims(spmdinfo.second[0], {});

    VLOG(4) << "Test CrossEntropyWithSoftmaxInferSpmd sharding on other axes."
            << std::endl
            << std::endl
            << std::endl;
  }

  // backward
  {
    std::vector<int64_t> loss_shape = {32, 1};
    // Sharding along softmax axis.
    x_dist_attr.set_dims_mapping(std::vector<int64_t>{0, 1});
    label_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1}));
    auto label = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape),
                                                  label_dist_attr);
    auto softmax =
        phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);

    auto loss_dist_attr = x_dist_attr;
    loss_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));
    auto loss_grad = phi::distributed::DistMetaTensor(
        phi::make_ddim(loss_shape), x_dist_attr);

    int axis = 1;
    auto spmdinfo = CrossEntropyWithSoftmaxGradInferSpmd(
        label, softmax, loss_grad, true, true, true, 1, axis);

    EXPECT_EQ(spmdinfo.first.size(), 3UL);
    EXPECT_EQ(spmdinfo.second.size(), 1UL);
    check_dim_mapping(spmdinfo.first[0], {0, -1});
    check_dim_mapping(spmdinfo.first[1], {0, -1});
    check_dim_mapping(spmdinfo.first[2], {0, -1});
    check_dim_mapping(spmdinfo.second[0], {0, -1});
    check_partial_dims(spmdinfo.second[0], {});

    VLOG(4)
        << "Test CrossEntropyWithSoftmaxGradInferSpmd sharding on softmax axis."
        << std::endl
        << std::endl
        << std::endl;
  }
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
