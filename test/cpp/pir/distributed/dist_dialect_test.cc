// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <gtest/gtest.h>
#include <iostream>

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"

using namespace paddle::dialect;  // NOLINT

TEST(process_mesh_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  std::vector<std::string> dim_names_2 = {"x", "s"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);

  // construct a ProcessMeshAttribute.
  auto mesh_attr =
      ProcessMeshAttribute::get(ctx, mesh_shape, process_ids, dim_names);
  auto mesh_attr_1 = ProcessMeshAttribute::get(ctx, process_mesh);
  auto mesh_attr_2 =
      ProcessMeshAttribute::get(ctx, mesh_shape, process_ids, dim_names_2);
  EXPECT_EQ(mesh_attr, mesh_attr_1);
  EXPECT_NE(mesh_attr, mesh_attr_2);

  // test member function.
  EXPECT_EQ(mesh_attr.process_mesh(), process_mesh);
  EXPECT_EQ(mesh_attr.shape(), mesh_shape);
  EXPECT_EQ(mesh_attr.process_ids(), process_ids);
  EXPECT_EQ(mesh_attr.dim_names(), dim_names);
  EXPECT_EQ(mesh_attr.size(), 4);
  EXPECT_EQ(mesh_attr.ndim(), 2);
  EXPECT_EQ(mesh_attr.dim_size(0), 2);
  EXPECT_EQ(mesh_attr.dim_size("y"), 2);
  EXPECT_FALSE(mesh_attr.empty());
  EXPECT_TRUE(mesh_attr.contains(3));
  EXPECT_EQ(mesh_attr.hash(), process_mesh.hash());
  EXPECT_EQ(mesh_attr.to_string(), process_mesh.to_string());
}
TEST(tensor_dist_attr_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  std::vector<int64_t> dims_mapping = {0, -1};
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status,
      partial_status_1{{1, phi::ReduceType::kRedSum}};

  auto mesh_attr =
      ProcessMeshAttribute::get(ctx, mesh_shape, process_ids, dim_names);

  // construct a TensorDistAttribute.
  auto tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, dims_mapping, partial_status);
  auto tensor_dist_attr_1 =
      TensorDistAttribute::get(ctx, process_mesh, dims_mapping, partial_status);
  auto tensor_dist_attr_2 = TensorDistAttribute::get(
      ctx, process_mesh, dims_mapping, partial_status_1);
  EXPECT_EQ(tensor_dist_attr, tensor_dist_attr_1);
  EXPECT_NE(tensor_dist_attr, tensor_dist_attr_2);

  // test member function.
  EXPECT_EQ(tensor_dist_attr.mesh_attr(), mesh_attr);
  EXPECT_EQ(tensor_dist_attr.process_mesh(), process_mesh);
  EXPECT_EQ(tensor_dist_attr.dims_mapping(), dims_mapping);
  EXPECT_EQ(tensor_dist_attr.partial_status(), partial_status);
}

TEST(dist_dense_tensor_type_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  auto mesh_attr = ProcessMeshAttribute::get(ctx, process_mesh);

  std::vector<int64_t> dims_mapping = {0, -1};
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status{
      {1, phi::ReduceType::kRedSum}};
  // construct a TensorDistAttribute.
  auto tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, dims_mapping, partial_status);

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  common::DDim dims = {2, 2};
  common::DataLayout data_layout = common::DataLayout::NCHW;
  pir::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::DenseTensorType dense_tensor_type = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  auto dist_densor_type =
      DistDenseTensorType::get(ctx, dense_tensor_type, tensor_dist_attr, dims);

  EXPECT_EQ(dist_densor_type.process_mesh(), process_mesh);
  EXPECT_EQ(dist_densor_type.dims_mapping(), dims_mapping);
  EXPECT_EQ(dist_densor_type.partial_status(), partial_status);
  EXPECT_EQ(dist_densor_type.dtype().isa<pir::Float32Type>(), true);
  EXPECT_EQ(dist_densor_type.global_ddim(), dims);
  EXPECT_EQ(dist_densor_type.data_layout(), data_layout);
  EXPECT_EQ(dist_densor_type.local_ddim(), dims);
}
