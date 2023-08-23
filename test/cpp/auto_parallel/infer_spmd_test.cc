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

#include <iostream>
#include <sstream>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/infermeta/distributed/binary.h"

namespace phi {
namespace distributed {
namespace test {

TEST(MatmulSPMDRule, Ctor) {
  // build input data class
  auto x_shape = DDim({64, 32});
  auto y_shape = DDim({32, 48});

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  auto dist_x = phi::distributed::DistTensor(x_shape, x_dist_attr);
  auto dist_y = phi::distributed::DistTensor(y_shape, y_dist_attr);

  phi::MetaTensor x(dist_x);
  phi::MetaTensor y(dist_y);

  size_t input_size = 2;
  size_t output_size = 1;

  // mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]

  // dynamic infer spmd
  auto infered_dist_attrs_dy = phi::distributed::MatmulInferSpmd(
      x, y, /*trans_x=*/false, /*trans_y=*/false);

  EXPECT_EQ(infered_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_dy.second.size(), output_size);

  EXPECT_EQ(infered_dist_attrs_dy.first[0].dims_mapping(),
            std::vector<int64_t>({1, -1}));
  EXPECT_EQ(infered_dist_attrs_dy.first[1].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs_dy.second[0].dims_mapping(),
            std::vector<int64_t>({1, -1}));
  EXPECT_EQ(infered_dist_attrs_dy.second[0].is_partial(), false);
  VLOG(4) << "test1 dynamic done.";

  // static infer spmd
  phi::distributed::InferSpmdContext ctx;
  ctx.EmplaceBackInput(x);
  ctx.EmplaceBackInput(y);
  ctx.EmplaceBackAttr(/*trans_x=*/false);
  ctx.EmplaceBackAttr(/*trans_x=*/false);
  auto matmul_spmd_rule =
      phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule("matmul");
  auto infered_dist_attrs_st = matmul_spmd_rule.InferForward(ctx);

  EXPECT_EQ(infered_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_st.second.size(), output_size);

  EXPECT_EQ(infered_dist_attrs_st.first[0].dims_mapping(),
            std::vector<int64_t>({1, -1}));
  EXPECT_EQ(infered_dist_attrs_st.first[1].dims_mapping(),
            std::vector<int64_t>({-1, -1}));
  EXPECT_EQ(infered_dist_attrs_st.second[0].dims_mapping(),
            std::vector<int64_t>({1, -1}));
  EXPECT_EQ(infered_dist_attrs_st.second[0].is_partial(), false);
  VLOG(4) << "test1 static done.";
}

}  // namespace test
}  // namespace distributed
}  // namespace phi
