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
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/infermeta/spmd_rules/replicated.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

auto& get_dims_mapping(const phi::distributed::ArgDistAttr& dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr));
  const auto& tensor_attr = paddle::get<0>(dist_attr);
  return tensor_attr.dims_mapping();
}

bool is_partial(const phi::distributed::ArgDistAttr& dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr));
  const auto& tensor_attr = paddle::get<0>(dist_attr);
  return tensor_attr.is_partial();
}

auto get_partial_dims(const phi::distributed::ArgDistAttr& dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr));
  const auto& tensor_attr = paddle::get<0>(dist_attr);
  return tensor_attr.partial_dims();
}

void check_dim_mapping(const phi::distributed::ArgDistAttr& dist_attr,
                       const std::vector<int64_t>& dim_mapping,
                       const std::string& line = "") {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr))
      << line;
  EXPECT_EQ(get_dims_mapping(dist_attr), dim_mapping) << line;
}

void check_partial_dims(const phi::distributed::ArgDistAttr& dist_attr,
                        const std::set<int64_t>& dims,
                        const std::string& line = "") {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr))
      << line;
  EXPECT_EQ(get_partial_dims(dist_attr), dims) << line;
}

void clean_partial_status(phi::distributed::ArgDistAttr* dist_attr) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(*dist_attr));
  auto& tensor_attr = paddle::get<0>(*dist_attr);
  tensor_attr.clean_partial_status();
}

void clean_partial_dims(phi::distributed::ArgDistAttr* dist_attr,
                        std::vector<int64_t> dims) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(*dist_attr));
  auto& tensor_attr = paddle::get<0>(*dist_attr);
  tensor_attr.clean_partial_dims(dims);
}

void set_partial_status(phi::distributed::ArgDistAttr* dist_attr,
                        std::vector<int64_t> dims) {
  EXPECT_TRUE(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(*dist_attr));
  auto& tensor_attr = paddle::get<0>(*dist_attr);
  tensor_attr.set_partial_status(dims);
}

TEST(MatmulSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 32};
  std::vector<int64_t> y_shape = {32, 48};

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

  size_t input_size = 2;
  size_t output_size = 1;

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(phi::make_ddim(y_shape), y_dist_attr);

  auto matmul_spmd_rule =
      phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule("matmul");

  // mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]
  phi::distributed::InferSpmdContext ctx(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  auto infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);

  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  check_dim_mapping(infered_dist_attrs.first[0], {1, -1});
  check_dim_mapping(infered_dist_attrs.first[1], {-1, -1});
  check_dim_mapping(infered_dist_attrs.second[0], {1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), false);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;

  // mk[-1,-1],kn[-1,0] --> mk[-1,-1],kn[-1,0] = nm[-1,0] partial[]
  x_dist_attr.set_dims_mapping({-1, -1});
  y_dist_attr.set_dims_mapping({-1, 0});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(infered_dist_attrs.first[0], {-1, -1});
  check_dim_mapping(infered_dist_attrs.first[1], {-1, 0});
  check_dim_mapping(infered_dist_attrs.second[0], {-1, 0});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), false);
  VLOG(4) << "test2 done." << std::endl << std::endl << std::endl;
  // mk[1, 0],kn[-1,-1] --> mk[1, 0],kn[0, -1] = nm[1, -1] partial[0]: done
  x_dist_attr.set_dims_mapping({1, 0});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(infered_dist_attrs.first[0], {1, 0});
  check_dim_mapping(infered_dist_attrs.first[1], {0, -1});
  check_dim_mapping(infered_dist_attrs.second[0], {1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), true);
  check_partial_dims(infered_dist_attrs.second[0], {0});
  VLOG(4) << "test3 done." << std::endl << std::endl << std::endl;

  // mk[-1,-1],kn[1,0] --> mk[-1, 1],kn[1, 0] = nm[-1, 0] partial[1]: done
  x_dist_attr.set_dims_mapping({-1, -1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(infered_dist_attrs.first[0], {-1, 1});
  check_dim_mapping(infered_dist_attrs.first[1], {1, 0});
  check_dim_mapping(infered_dist_attrs.second[0], {-1, 0});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), true);
  check_partial_dims(infered_dist_attrs.second[0], {1});
  VLOG(4) << "test4 done." << std::endl << std::endl << std::endl;

  // abcmk[1, 0, -1, -1],kn[-1, -1] --> abcmk[1, 0, -1, -1],kn[-1, -1] =
  // abcmn[1, 0, -1, -1] partial[]: done
  x_shape = {512, 48, 64, 32};
  x_dist_attr.set_dims_mapping({0, 1, -1, -1});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(infered_dist_attrs.first[0], {0, 1, -1, -1});
  check_dim_mapping(infered_dist_attrs.first[1], {-1, -1});
  check_dim_mapping(infered_dist_attrs.second[0], {0, 1, -1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), false);
  VLOG(4) << "test5 done." << std::endl << std::endl << std::endl;

  // abcmk[1, -1, -1, 0],kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[0, -1] = abcmn[1,
  // -1, -1, -1] partial[0]: done
  x_dist_attr.set_dims_mapping({1, -1, -1, 0});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/false});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(infered_dist_attrs.first[0], {1, -1, -1, 0});
  check_dim_mapping(infered_dist_attrs.first[1], {0, -1});
  check_dim_mapping(infered_dist_attrs.second[0], {1, -1, -1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), true);
  check_partial_dims(infered_dist_attrs.second[0], {0});
  VLOG(4) << "test6 done." << std::endl << std::endl << std::endl;

  // abcmk[1, -1, -1, 0], kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[-1, -1] =
  // abcmn[1, -1, 0, -1] partial[]: done
  x_dist_attr.set_dims_mapping({1, -1, -1, 0});
  y_dist_attr.set_dims_mapping({-1, -1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/false});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);

  check_dim_mapping(infered_dist_attrs.first[0], {1, -1, -1, 0});
  check_dim_mapping(infered_dist_attrs.first[1], {-1, -1});
  check_dim_mapping(infered_dist_attrs.second[0], {1, -1, 0, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), false);

  VLOG(4) << "test7 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, -1, -1], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]: done
  x_dist_attr.set_dims_mapping({-1, -1, -1, -1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/false, /*trans_x=*/true});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(infered_dist_attrs.first[0], {-1, -1, -1, 0});
  check_dim_mapping(infered_dist_attrs.first[1], {1, 0});
  check_dim_mapping(infered_dist_attrs.second[0], {-1, -1, -1, 1});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), true);
  check_partial_dims(infered_dist_attrs.second[0], {0});
  clean_partial_dims(&infered_dist_attrs.second[0], {0});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), false);
  VLOG(4) << "test8 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, 0, 1]+trans_x=true, kn[1, 0]+trans_y=true --> abcmk[-1, -1,
  // 0, -1],kn[-1, 0] = abcmn[-1, -1, 1, -1] partial[0]: done
  x_dist_attr.set_dims_mapping({-1, -1, 0, 1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/true});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);

  check_dim_mapping(infered_dist_attrs.first[0], {-1, -1, 0, 1});
  check_dim_mapping(infered_dist_attrs.first[1],
                    {-1, 0});  // confilct and should be changed to [-1, 0]
  check_dim_mapping(infered_dist_attrs.second[0], {-1, -1, 1, -1});
  check_partial_dims(infered_dist_attrs.second[0], {0});

  clean_partial_status(&infered_dist_attrs.second[0]);
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), false);
  EXPECT_ANY_THROW(set_partial_status(&infered_dist_attrs.second[0], {1}));
  VLOG(4) << "test9 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, 1, 0], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] =
  // abcmn[-1, -1, -1, 1] partial[0]: done
  x_dist_attr.set_dims_mapping({-1, -1, 1, 0});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/true});
  EXPECT_ANY_THROW(infered_dist_attrs = matmul_spmd_rule.InferForward(ctx));
  // Error
  VLOG(4) << "test10 done." << std::endl << std::endl << std::endl;

  // abcmk[-1, -1, 1, 0], kn[0, 1] --> abcmk[-1, -1, 1, 0],kn[0, 1] =
  // abcmn[-1, -1, 1, -1] partial[0]:
  x_dist_attr.set_dims_mapping({-1, -1, 0, 1});
  y_dist_attr.set_dims_mapping({1, 0});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  ctx = phi::distributed::InferSpmdContext(
      {x, y}, {/*trans_x=*/true, /*trans_x=*/true});
  infered_dist_attrs = matmul_spmd_rule.InferForward(ctx);
  check_dim_mapping(infered_dist_attrs.second[0], {-1, -1, 1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), true);
  check_partial_dims(infered_dist_attrs.second[0], {0});

  // try to clean partial on a dim which is not partial
  EXPECT_ANY_THROW(clean_partial_dims(&infered_dist_attrs.second[0], {1}));
  // try to clean partial on a dims which is sharded
  EXPECT_ANY_THROW(set_partial_status(&infered_dist_attrs.second[0], {1}));

  // clean partial and then re-set again
  clean_partial_dims(&infered_dist_attrs.second[0], {0});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), false);
  set_partial_status(&infered_dist_attrs.second[0], {0});
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), true);
  check_partial_dims(infered_dist_attrs.second[0], {0});
  VLOG(4) << "test11 done." << std::endl << std::endl << std::endl;
}

TEST(LayerNormSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {64, 32, 1024};
  std::vector<int64_t> scale_shape = {1024};
  std::vector<int64_t> bias_shape = {1024};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false, false}));

  TensorDistAttr scale_dist_attr = TensorDistAttr();
  scale_dist_attr.set_process_mesh(process_mesh);
  scale_dist_attr.set_dims_mapping(std::vector<int64_t>({-1}));
  scale_dist_attr.set_dynamic_dims(std::vector<bool>({false}));

  TensorDistAttr bias_dist_attr = TensorDistAttr();
  bias_dist_attr.set_process_mesh(process_mesh);
  bias_dist_attr.set_dims_mapping(std::vector<int64_t>({-1}));
  bias_dist_attr.set_dynamic_dims(std::vector<bool>({false}));

  paddle::framework::AttributeMap attrs;
  float epsilon = 1e-5;
  int begin_norm_axis = 2;

  auto layer_norm_rule =
      phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule("layer_norm");

  // ijk[1, -1, -1], k[-1], k[-1] --> ijk[1, -1, -1], z[1], z[1], z=ij,
  // begin_norm_axis=2
  begin_norm_axis = 2;
  x_dist_attr.set_dims_mapping({1, -1, -1});
  scale_dist_attr.set_dims_mapping({-1});
  bias_dist_attr.set_dims_mapping({-1});
  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor scale(phi::make_ddim(scale_shape),
                                         scale_dist_attr);
  phi::distributed::DistMetaTensor bias(phi::make_ddim(bias_shape),
                                        bias_dist_attr);
  phi::distributed::InferSpmdContext ctx({x, scale, bias},
                                         {epsilon, begin_norm_axis});
  auto infered_dist_attrs = layer_norm_rule.InferForward(ctx);

  size_t input_size = 3;
  size_t output_size = 3;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);
  check_dim_mapping(infered_dist_attrs.first[0], {1, -1, -1});
  check_dim_mapping(infered_dist_attrs.first[1], {-1});
  check_dim_mapping(infered_dist_attrs.first[2], {-1});
  check_dim_mapping(infered_dist_attrs.second[0], {1, -1, -1});
  check_dim_mapping(infered_dist_attrs.second[1], {1});
  check_dim_mapping(infered_dist_attrs.second[2], {1});
  VLOG(4) << "test1 done.";

  // ijk[1, 0, -1],k[0],k[0] --> ijk[1, -1, -1],z[1],z[1],
  // begin_norm_axis=2
  begin_norm_axis = 2;
  x_dist_attr.set_dims_mapping({1, 0, -1});
  scale_dist_attr.set_dims_mapping({0});
  bias_dist_attr.set_dims_mapping({0});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  scale = phi::distributed::DistMetaTensor(phi::make_ddim(scale_shape),
                                           scale_dist_attr);
  bias = phi::distributed::DistMetaTensor(phi::make_ddim(bias_shape),
                                          bias_dist_attr);
  ctx = phi::distributed::InferSpmdContext({x, scale, bias},
                                           {epsilon, begin_norm_axis});
  infered_dist_attrs = layer_norm_rule.InferForward(ctx);

  check_dim_mapping(infered_dist_attrs.first[0], {1, -1, -1});
  check_dim_mapping(infered_dist_attrs.first[1], {-1});
  check_dim_mapping(infered_dist_attrs.first[2], {-1});
  check_dim_mapping(infered_dist_attrs.second[0], {1, -1, -1});
  check_dim_mapping(infered_dist_attrs.second[1], {1});
  check_dim_mapping(infered_dist_attrs.second[2], {1});
  VLOG(4) << "test2 done.";

  // ijk[0, -1, -1],y[-1],y[1] --> ijk[0, 1, -1], i[0], i[0], y=jk,
  // begin_norm_axis=1
  begin_norm_axis = 1;
  x_dist_attr.set_dims_mapping({0, -1, -1});
  scale_dist_attr.set_dims_mapping({-1});
  bias_dist_attr.set_dims_mapping({1});
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  scale = phi::distributed::DistMetaTensor(phi::make_ddim(scale_shape),
                                           scale_dist_attr);
  bias = phi::distributed::DistMetaTensor(phi::make_ddim(bias_shape),
                                          bias_dist_attr);
  ctx = phi::distributed::InferSpmdContext({x, scale, bias},
                                           {epsilon, begin_norm_axis});
  infered_dist_attrs = layer_norm_rule.InferForward(ctx);

  check_dim_mapping(infered_dist_attrs.first[0], {0, -1, -1});
  check_dim_mapping(infered_dist_attrs.first[1], {-1});
  check_dim_mapping(infered_dist_attrs.first[2], {-1});
  check_dim_mapping(infered_dist_attrs.second[0], {0, -1, -1});
  check_dim_mapping(infered_dist_attrs.second[1], {0});
  check_dim_mapping(infered_dist_attrs.second[2], {0});
  VLOG(4) << "test3 done.";
}

TEST(MatmulSPMDRuleInferBackward, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {512, 1024, 64, 32};
  std::vector<int64_t> y_shape = {512, 1, 32, 48};
  std::vector<int64_t> out_shape = {512, 1024, 64, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(
      std::vector<int64_t>({-1, 1, 0, -1}));  // no affect
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(
      std::vector<int64_t>({0, 1, -1, -1}));  // no affect
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out_dist_attr = TensorDistAttr();
  out_dist_attr.set_process_mesh(process_mesh);
  out_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 1, -1}));
  out_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));
  out_dist_attr.set_partial_status(std::vector<int64_t>({0}));

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(phi::make_ddim(y_shape), y_dist_attr);
  phi::distributed::DistMetaTensor out(phi::make_ddim(out_shape),
                                       out_dist_attr);

  auto matmul_spmd_rule =
      phi::distributed::SpmdRuleFactory::Instance().GetSpmdRule("matmul");

  // TODO(zyc) update in future: propogate the partial in inferbackward
  // abmn[-1, -1, 1, -1] + partial[0] --> abmk[-1, -1, 1, -1], a1kn[-1, -1, -1,
  // -1]
  phi::distributed::InferSpmdContext ctx(
      {x, y, out}, {/*trans_x=*/false, /*trans_x=*/false});
  auto infered_dist_attrs = matmul_spmd_rule.InferBackward(ctx);

  size_t input_size = 2;
  size_t output_size = 1;
  EXPECT_EQ(infered_dist_attrs.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs.second.size(), output_size);

  check_dim_mapping(infered_dist_attrs.first[0], {-1, -1, 1, -1});
  check_dim_mapping(infered_dist_attrs.first[1], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs.second[0], {-1, -1, 1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs.first[0]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs.first[1]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs.second[0]), true);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;
}

TEST(ReplicatedSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {10, 10, 32, 48};
  std::vector<int64_t> y_shape = {32, 48};
  std::vector<int64_t> out1_shape = {10, 10, 32, 48};
  std::vector<int64_t> out2_shape = {10, 32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1, -1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1}));  // no affect
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out1_dist_attr = TensorDistAttr();
  out1_dist_attr.set_process_mesh(process_mesh);
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 1, -1}));
  out1_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out2_dist_attr = TensorDistAttr();
  out2_dist_attr.set_process_mesh(process_mesh);
  out2_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1}));
  out2_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(phi::make_ddim(y_shape), y_dist_attr);
  phi::distributed::DistMetaTensor out1(phi::make_ddim(out1_shape),
                                        out1_dist_attr);
  phi::distributed::DistMetaTensor out2(phi::make_ddim(out2_shape),
                                        out2_dist_attr);

  // 2 inputs 2 outputs
  // call in vector arguments format
  auto infered_dist_attrs_st =
      phi::distributed::ReplicatedInferSpmd({&x, &y}, {&out1, &out2});
  // call in variadic arguments format
  auto infered_dist_attrs_dy =
      phi::distributed::VariadicReplicatedInferSpmd(x, y, &out1, &out2);

  size_t input_size = 2;
  size_t output_size = 2;
  EXPECT_EQ(infered_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_dy.second.size(), output_size);

  check_dim_mapping(infered_dist_attrs_st.first[0], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_st.first[1], {-1, -1});
  check_dim_mapping(infered_dist_attrs_st.second[0], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_st.second[1], {-1, -1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs_st.first[0]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs_st.first[1]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs_st.second[0]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs_st.second[1]), false);
  EXPECT_EQ(infered_dist_attrs_st.first, infered_dist_attrs_dy.first);
  EXPECT_EQ(infered_dist_attrs_st.second, infered_dist_attrs_dy.second);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;

  // 3 inputs 1 outputs
  // call in vector arguments format
  infered_dist_attrs_st =
      phi::distributed::ReplicatedInferSpmd({&x, &y, &out1}, {&out2});
  // call in variadic arguments format
  infered_dist_attrs_dy =
      phi::distributed::VariadicReplicatedInferSpmd(x, y, out1, &out2);

  input_size = 3;
  output_size = 1;
  EXPECT_EQ(infered_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_dy.second.size(), output_size);
  check_dim_mapping(infered_dist_attrs_dy.first[0], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.first[1], {-1, -1});
  check_dim_mapping(infered_dist_attrs_dy.first[2], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[0], {-1, -1, -1});
  EXPECT_EQ(infered_dist_attrs_st.first, infered_dist_attrs_dy.first);
  EXPECT_EQ(infered_dist_attrs_st.second, infered_dist_attrs_dy.second);
  VLOG(4) << "test2 done." << std::endl << std::endl << std::endl;

  // 1 inputs 3 outputs backward
  // call in vector arguments format
  infered_dist_attrs_st =
      phi::distributed::ReplicatedInferSpmdReverse({&x}, {&y, &out1, &out2});
  // call in variadic arguments format
  infered_dist_attrs_dy =
      phi::distributed::VariadicReplicatedInferSpmdReverse(x, &y, &out1, &out2);

  input_size = 1;
  output_size = 3;
  EXPECT_EQ(infered_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_dy.second.size(), output_size);

  check_dim_mapping(infered_dist_attrs_dy.first[0], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[0], {-1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[1], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[2], {-1, -1, -1});
  EXPECT_EQ(infered_dist_attrs_st.first, infered_dist_attrs_dy.first);
  EXPECT_EQ(infered_dist_attrs_st.second, infered_dist_attrs_dy.second);
  VLOG(4) << "test3 done." << std::endl << std::endl << std::endl;
}

TEST(DefaultDataParallelSPMDRule, Ctor) {
  // build input data class
  std::vector<int64_t> x_shape = {10, 10, 32, 48};
  std::vector<int64_t> y_shape = {32, 48};
  std::vector<int64_t> out1_shape = {10, 10, 32, 48};
  std::vector<int64_t> out2_shape = {10, 32, 48};

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  TensorDistAttr x_dist_attr = TensorDistAttr();
  x_dist_attr.set_process_mesh(process_mesh);
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, -1, 1}));
  x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr y_dist_attr = TensorDistAttr();
  y_dist_attr.set_process_mesh(process_mesh);
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1}));
  y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out1_dist_attr = TensorDistAttr();
  out1_dist_attr.set_process_mesh(process_mesh);
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1, 1, -1}));
  out1_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  TensorDistAttr out2_dist_attr = TensorDistAttr();
  out2_dist_attr.set_process_mesh(process_mesh);
  out2_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 1, -1}));
  out2_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

  phi::distributed::DistMetaTensor x(phi::make_ddim(x_shape), x_dist_attr);
  phi::distributed::DistMetaTensor y(phi::make_ddim(y_shape), y_dist_attr);
  phi::distributed::DistMetaTensor out1(phi::make_ddim(out1_shape),
                                        out1_dist_attr);
  phi::distributed::DistMetaTensor out2(phi::make_ddim(out2_shape),
                                        out2_dist_attr);

  // 2 inputs 2 outputs, batch axis sharding is propagatd while other axes are
  // replicatd call in vector arguments format
  auto infered_dist_attrs_st =
      phi::distributed::DefaultDataParallelInferSpmd({&x, &y}, {&out1, &out2});
  // call in variadic arguments format
  auto infered_dist_attrs_dy =
      phi::distributed::VariadicDefaultDataParallelInferSpmd(
          x, y, &out1, &out2);

  size_t input_size = 2;
  size_t output_size = 2;
  EXPECT_EQ(infered_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_dy.second.size(), output_size);
  check_dim_mapping(infered_dist_attrs_st.first[0], {0, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_st.first[1], {0, -1});
  check_dim_mapping(infered_dist_attrs_st.second[0], {0, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_st.second[1], {0, -1, -1});
  EXPECT_EQ(is_partial(infered_dist_attrs_st.first[0]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs_st.first[1]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs_st.second[0]), false);
  EXPECT_EQ(is_partial(infered_dist_attrs_st.second[1]), false);

  EXPECT_EQ(infered_dist_attrs_st.first, infered_dist_attrs_dy.first);
  EXPECT_EQ(infered_dist_attrs_st.second, infered_dist_attrs_dy.second);
  VLOG(4) << "test1 done." << std::endl << std::endl << std::endl;

  // 1 inputs 3 outputs, batch axis is un-sharded
  // call in vector arguments format
  infered_dist_attrs_st =
      phi::distributed::DefaultDataParallelInferSpmd({&x}, {&y, &out1, &out2});
  // call in variadic arguments format
  infered_dist_attrs_dy =
      phi::distributed::VariadicDefaultDataParallelInferSpmd(
          x, &y, &out1, &out2);

  input_size = 1;
  output_size = 3;
  EXPECT_EQ(infered_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_dy.second.size(), output_size);

  check_dim_mapping(infered_dist_attrs_dy.first[0], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[0], {-1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[1], {-1, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[2], {-1, -1, -1});

  EXPECT_EQ(infered_dist_attrs_st.first, infered_dist_attrs_dy.first);
  EXPECT_EQ(infered_dist_attrs_st.second, infered_dist_attrs_dy.second);
  VLOG(4) << "test2 done." << std::endl << std::endl << std::endl;

  // conflict on batch axis
  x_dist_attr.set_dims_mapping(std::vector<int64_t>({0, -1, -1, -1}));
  y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1, -1, -1}));
  x = phi::distributed::DistMetaTensor(phi::make_ddim(x_shape), x_dist_attr);
  y = phi::distributed::DistMetaTensor(phi::make_ddim(y_shape), y_dist_attr);
  out1 = phi::distributed::DistMetaTensor(phi::make_ddim(out1_shape),
                                          out1_dist_attr);

  EXPECT_ANY_THROW(infered_dist_attrs_st =
                       phi::distributed::DefaultDataParallelInferSpmd(
                           {&x, &y, &out1}, {&out2}));
  // call in variadic arguments format
  EXPECT_ANY_THROW(infered_dist_attrs_dy =
                       phi::distributed::VariadicDefaultDataParallelInferSpmd(
                           x, y, out1, &out2));

  VLOG(4) << "test3 done." << std::endl << std::endl << std::endl;

  // 2 inputs 2 outputs, backward
  // call in vector arguments format
  out1_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, 0, 1, -1}));
  out2_dist_attr.set_dims_mapping(std::vector<int64_t>({0, 1, -1}));
  out1 = phi::distributed::DistMetaTensor(phi::make_ddim(out1_shape),
                                          out1_dist_attr);
  out2 = phi::distributed::DistMetaTensor(phi::make_ddim(out2_shape),
                                          out2_dist_attr);

  infered_dist_attrs_st = phi::distributed::DefaultDataParallelInferSpmdReverse(
      {&x, &y}, {&out1, &out2});
  // call in variadic arguments format
  infered_dist_attrs_dy =
      phi::distributed::VariadicDefaultDataParallelInferSpmdReverse(
          x, y, &out1, &out2);

  input_size = 2;
  output_size = 2;
  EXPECT_EQ(infered_dist_attrs_st.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_st.second.size(), output_size);
  EXPECT_EQ(infered_dist_attrs_dy.first.size(), input_size);
  EXPECT_EQ(infered_dist_attrs_dy.second.size(), output_size);
  check_dim_mapping(infered_dist_attrs_dy.first[0], {0, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.first[1], {0, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[0], {0, -1, -1, -1});
  check_dim_mapping(infered_dist_attrs_dy.second[1], {0, -1, -1});
  EXPECT_EQ(infered_dist_attrs_st.first, infered_dist_attrs_dy.first);
  EXPECT_EQ(infered_dist_attrs_st.second, infered_dist_attrs_dy.second);
  VLOG(4) << "test4 done." << std::endl << std::endl << std::endl;
}
TEST(ConcatRule, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<std::vector<int64_t>> shapes = {
      {16, 16, 16}, {4, 16, 16}, {2, 16, 16}};
  std::vector<std::vector<int64_t>> dim_mappings = {
      {-1, 0, 1}, {-1, 1, 0}, {-1, -1, 0}};
  std::vector<std::vector<int64_t>> partial_status = {{}, {}, {1}};

  auto build_inputs = [&] {
    std::vector<phi::distributed::DistMetaTensor> inputs;
    for (int i = 0; i < 3; i++) {
      auto t_dist_attr = TensorDistAttr();
      t_dist_attr.set_process_mesh(process_mesh);
      t_dist_attr.set_dims_mapping(dim_mappings[i]);
      t_dist_attr.set_dynamic_dims({false, false, false});
      auto input = phi::distributed::DistMetaTensor(phi::make_ddim(shapes[i]),
                                                    t_dist_attr);
      inputs.push_back(input);
    }
    return inputs;
  };

  // test 1, inputs are aligned according to cost, and partial status is cleared
  auto inputs = build_inputs();
  auto infered_dist_attrs = phi::distributed::ConcatInferSpmd(inputs, 0);
  // list of tensor => sigle tensor
  EXPECT_EQ(infered_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(infered_dist_attrs.second.size(), static_cast<size_t>(1));
  EXPECT_TRUE(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          infered_dist_attrs.first[0]));
  EXPECT_TRUE(paddle::holds_alternative<phi::distributed::TensorDistAttr>(
      infered_dist_attrs.second[0]));
  auto& inputs_infer1 = paddle::get<1>(infered_dist_attrs.first[0]);
  for (auto e : inputs_infer1) {
    check_dim_mapping(e, {-1, 1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(infered_dist_attrs.second[0], {-1, 1, 0});
  check_partial_dims(infered_dist_attrs.second[0], {});

  // test 2，force replicate along concat axis
  inputs = build_inputs();
  infered_dist_attrs = phi::distributed::ConcatInferSpmd(inputs, 1);
  // list of tensor => sigle tensor
  EXPECT_EQ(infered_dist_attrs.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(infered_dist_attrs.second.size(), static_cast<size_t>(1));
  EXPECT_TRUE(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          infered_dist_attrs.first[0]));
  EXPECT_TRUE(paddle::holds_alternative<phi::distributed::TensorDistAttr>(
      infered_dist_attrs.second[0]));
  auto& inputs_infer2 = paddle::get<1>(infered_dist_attrs.first[0]);
  for (auto e : inputs_infer2) {
    check_dim_mapping(e, {1, -1, 0});
    check_partial_dims(e, {});
  }
  check_dim_mapping(infered_dist_attrs.second[0], {1, -1, 0});
  check_partial_dims(infered_dist_attrs.second[0], {});
}

TEST(LayerNorm, Ctor) {
  using phi::distributed::PartialStatus;
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> x_shapes = {16, 32, 32};

  auto build_input = [&](const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& dim_mapping) {
    auto t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(dim_mapping);
    t_dist_attr.set_dynamic_dims({false, false, false});
    auto input =
        phi::distributed::DistMetaTensor(phi::make_ddim(shape), t_dist_attr);
    return input;
  };
  // test 1
  auto x = build_input(x_shapes, {0, 1, -1});
  auto out_grad = build_input(x_shapes, {0, 1, -1});
  auto mean = build_input({16, 32}, {0, 1});
  auto variance = build_input({16, 32}, {0, 1});
  auto scale = build_input({32}, {0});
  auto bias = build_input({32}, {0});

  auto spmd1 =
      LayerNormGradInferSpmd(x, mean, variance, scale, bias, out_grad, 1.0, 2);

  // test 2
  mean = build_input({16}, {0});
  variance = build_input({16}, {0});
  scale = build_input({32, 32}, {0, 1});
  bias = build_input({32, 32}, {0, 1});
  auto spmd2 =
      LayerNormGradInferSpmd(x, mean, variance, scale, bias, out_grad, 1.0, 1);
}

TEST(Util, Ctor) {
  // test equal test not equal
  using phi::distributed::PartialStatus;
  using phi::distributed::PlacementEqual;
  using phi::distributed::ReplicatedStatus;
  using phi::distributed::ShardStatus;
  auto a = std::make_shared<PartialStatus>(phi::ReduceType::kRedSum);
  auto b = std::make_shared<PartialStatus>(phi::ReduceType::kRedMin);
  EXPECT_TRUE(PlacementEqual(a, a));
  EXPECT_TRUE(!PlacementEqual(a, b));
  auto c = std::make_shared<ShardStatus>(0);
  auto d = std::make_shared<ShardStatus>(1);
  EXPECT_TRUE(!PlacementEqual(a, c));
  EXPECT_TRUE(!PlacementEqual(b, c));
  EXPECT_TRUE(PlacementEqual(c, c));
  EXPECT_TRUE(!PlacementEqual(c, d));
  auto e = std::make_shared<ReplicatedStatus>();
  EXPECT_TRUE(PlacementEqual(e, e));
  EXPECT_TRUE(!PlacementEqual(a, e));
  EXPECT_TRUE(!PlacementEqual(b, e));
  EXPECT_TRUE(!PlacementEqual(c, e));
  EXPECT_TRUE(!PlacementEqual(d, e));
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
