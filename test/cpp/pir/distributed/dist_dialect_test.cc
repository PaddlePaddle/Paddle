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
#include "paddle/fluid/pir/dialect/distributed/ir/dist_interface.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/program.h"

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
  EXPECT_EQ(tensor_dist_attr.process_mesh_attr(), mesh_attr);
  EXPECT_EQ(tensor_dist_attr.process_mesh_attr().process_mesh(), process_mesh);
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
  pir::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::DenseTensorType dense_tensor_type = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  auto dist_densor_type =
      DistDenseTensorType::get(ctx, dense_tensor_type, tensor_dist_attr, dims);

  EXPECT_EQ(dist_densor_type.process_mesh_attr(), mesh_attr);
  EXPECT_EQ(dist_densor_type.process_mesh_attr().process_mesh(), process_mesh);
  EXPECT_EQ(dist_densor_type.dims_mapping(), dims_mapping);
  EXPECT_EQ(dist_densor_type.partial_status(), partial_status);
  EXPECT_EQ(dist_densor_type.dtype().isa<pir::Float32Type>(), true);
  EXPECT_EQ(dist_densor_type.global_ddim(), dims);
  EXPECT_EQ(dist_densor_type.data_layout(), data_layout);
  EXPECT_EQ(dist_densor_type.local_ddim(), dims);
}

TEST(dist_dense_tensor_type_test, warp_type_interface) {
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
  pir::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::DenseTensorType dense_tensor_type = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  pir::Type dist_densor_type =
      DistDenseTensorType::get(ctx, dense_tensor_type, tensor_dist_attr, dims);

  EXPECT_TRUE(dist_densor_type.isa<pir::DenseTensorType>());
  EXPECT_EQ(dist_densor_type.dyn_cast<pir::DenseTensorType>(),
            dense_tensor_type);
}

TEST(dist_dense_tensor_type_test, dist_interface) {
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
  common::DDim dims = {4, 8};
  common::DDim local_dims = {2, 8};
  common::DataLayout data_layout = common::DataLayout::NCHW;
  pir::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::DenseTensorType dense_tensor_type = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  pir::Type dist_densor_type =
      DistDenseTensorType::get(ctx, dense_tensor_type, tensor_dist_attr);

  EXPECT_TRUE(dist_densor_type.isa<pir::DenseTensorType>());
  EXPECT_EQ(dist_densor_type.dyn_cast<pir::DenseTensorType>(),
            dense_tensor_type);

  // test local cast
  auto local_dense_tensor_type = dist_densor_type.dyn_cast<DistTypeInterface>()
                                     .local_type()
                                     .dyn_cast<pir::DenseTensorType>();
  EXPECT_TRUE(local_dense_tensor_type.isa<pir::DenseTensorType>());
  EXPECT_FALSE(local_dense_tensor_type.isa<DistDenseTensorType>());
  EXPECT_EQ(local_dense_tensor_type.dtype().isa<pir::Float32Type>(), true);
  EXPECT_EQ(local_dense_tensor_type.dims(), local_dims);
  EXPECT_EQ(local_dense_tensor_type.data_layout(), data_layout);
  EXPECT_EQ(local_dense_tensor_type.lod(), lod);
  EXPECT_EQ(local_dense_tensor_type.offset(), offset);
}

TEST(operation_dist_attr_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status;

  auto mesh_attr =
      ProcessMeshAttribute::get(ctx, mesh_shape, process_ids, dim_names);
  std::vector<int64_t> dims_mapping = {0, -1};

  // construct a OperationDistAttribute.
  auto x_tensor_dist_attr =
      TensorDistAttribute::get(ctx, process_mesh, dims_mapping, partial_status);
  auto y_tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, dims_mapping, partial_status);
  auto out_tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, dims_mapping, partial_status);

  auto operand_attrs =
      std::vector<pir::Attribute>{x_tensor_dist_attr, y_tensor_dist_attr};
  auto result_attrs = std::vector<pir::Attribute>{out_tensor_dist_attr};
  auto op_attr = OperationDistAttribute::get(
      ctx, process_mesh, operand_attrs, result_attrs);
  auto op_attr_1 =
      OperationDistAttribute::get(ctx, mesh_attr, operand_attrs, result_attrs);

  // construct another OperationDistAttribute.
  std::vector<std::string> dim_names_2 = {"x", "s"};
  auto mesh_attr_2 =
      ProcessMeshAttribute::get(ctx, mesh_shape, process_ids, dim_names_2);

  auto x_tensor_dist_attr_2 =
      TensorDistAttribute::get(ctx, mesh_attr_2, dims_mapping, partial_status);
  auto y_tensor_dist_attr_2 =
      TensorDistAttribute::get(ctx, mesh_attr_2, dims_mapping, partial_status);
  auto out_tensor_dist_attr_2 =
      TensorDistAttribute::get(ctx, mesh_attr_2, dims_mapping, partial_status);

  auto operand_attrs_2 =
      std::vector<pir::Attribute>{x_tensor_dist_attr_2, y_tensor_dist_attr_2};
  auto result_attrs_2 = std::vector<pir::Attribute>{out_tensor_dist_attr_2};
  auto op_attr_2 = OperationDistAttribute::get(
      ctx, mesh_attr_2, operand_attrs_2, result_attrs_2);

  // check
  EXPECT_EQ(op_attr, op_attr_1);
  EXPECT_NE(op_attr, op_attr_2);
  EXPECT_EQ(op_attr.process_mesh_attr(), mesh_attr);
  EXPECT_EQ(op_attr.process_mesh_attr().process_mesh(), process_mesh);
  EXPECT_EQ(op_attr.operands(), operand_attrs);
  EXPECT_EQ(op_attr.operand(0), operand_attrs.at(0));
  EXPECT_EQ(op_attr.operand(1), operand_attrs.at(1));
  EXPECT_EQ(op_attr.num_operands(), (uint32_t)2);

  EXPECT_EQ(op_attr.results(), result_attrs);
  EXPECT_EQ(op_attr.result(0), result_attrs.at(0));
  EXPECT_EQ(op_attr.num_results(), (uint32_t)1);
}

TEST(shard_tensor_op_replicate_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  auto mesh_attr = ProcessMeshAttribute::get(ctx, process_mesh);

  std::vector<int64_t> data_shape = {12, 6};
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status;

  // construct a replicated
  std::vector<int64_t> dims_mapping = {-1, -1};

  auto data_op = builder.Build<paddle::dialect::DataOp>(
      "w0", data_shape, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<int64_t> local_shape = {12, 6};
  auto tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, dims_mapping, partial_status);

  pir::AttributeMap attr_map = {{"tensor_dist_attr", tensor_dist_attr}};

  paddle::dialect::ShardTensorOp shard_op =
      builder.Build<paddle::dialect::ShardTensorOp>(data_op.result(0),
                                                    attr_map);

  EXPECT_TRUE(shard_op.out().type().isa<DistDenseTensorType>());
  auto op_out_type = shard_op.out().type().dyn_cast<DistDenseTensorType>();
  EXPECT_EQ(op_out_type.global_ddim(), phi::make_ddim(data_shape));
  EXPECT_EQ(op_out_type.local_ddim(), phi::make_ddim(local_shape));
  EXPECT_EQ(op_out_type.process_mesh_attr(), mesh_attr);
  EXPECT_EQ(op_out_type.dims_mapping(), dims_mapping);
  EXPECT_EQ(op_out_type.partial_dims().size(), (size_t)0);

  EXPECT_EQ(
      shard_op.attribute<OperationDistAttribute>("op_dist_attr").num_operands(),
      (uint32_t)0);

  EXPECT_EQ(
      shard_op.attribute<OperationDistAttribute>("op_dist_attr").num_results(),
      (uint32_t)1);
  EXPECT_EQ(shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            mesh_attr);

  // check reshard
  std::vector<int64_t> dst_mesh_shape = {3, 2};
  std::vector<int64_t> dst_dims_mapping = {-1, 0};

  phi::distributed::ProcessMesh dst_process_mesh(
      dst_mesh_shape, process_ids, dim_names);
  auto dst_mesh_attr = ProcessMeshAttribute::get(ctx, dst_process_mesh);
  auto dst_tensor_dist_attr = TensorDistAttribute::get(
      ctx, dst_mesh_attr, dst_dims_mapping, partial_status);
  paddle::dialect::ReshardOp reshard_op =
      builder.Build<paddle::dialect::ReshardOp>(shard_op.out(),
                                                dst_tensor_dist_attr);

  EXPECT_TRUE(reshard_op.result(0).type().isa<DistDenseTensorType>());
  auto dst_op_out_type =
      reshard_op.result(0).type().dyn_cast<DistDenseTensorType>();
  EXPECT_EQ(dst_op_out_type.global_ddim(), phi::make_ddim(data_shape));
  EXPECT_EQ(dst_op_out_type.local_ddim(), phi::make_ddim({12, 2}));
  EXPECT_EQ(dst_op_out_type.process_mesh_attr(), dst_mesh_attr);
  EXPECT_EQ(dst_op_out_type.dims_mapping(), dst_dims_mapping);
  EXPECT_EQ(dst_op_out_type.partial_dims().size(), (size_t)0);

  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_operands(),
            (uint32_t)1);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_results(),
            (uint32_t)1);
  phi::distributed::ProcessMesh flatten_process_mesh(
      {6}, process_ids, {"merged"});
  auto flatten_mesh_attr = ProcessMeshAttribute::get(ctx, flatten_process_mesh);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            flatten_mesh_attr);
}

TEST(shard_tensor_op_shard_row_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  auto mesh_attr = ProcessMeshAttribute::get(ctx, process_mesh);

  std::vector<int64_t> data_shape = {12, 6};
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status;

  // construct a row shard
  std::vector<int64_t> dims_mapping = {1, -1};
  auto data_op = builder.Build<paddle::dialect::DataOp>(
      "w1", data_shape, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<int64_t> local_shape = {4, 6};
  auto tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, dims_mapping, partial_status);

  pir::AttributeMap attr_map = {{"tensor_dist_attr", tensor_dist_attr}};

  paddle::dialect::ShardTensorOp shard_op =
      builder.Build<paddle::dialect::ShardTensorOp>(data_op.result(0),
                                                    attr_map);

  EXPECT_TRUE(shard_op.out().type().isa<DistDenseTensorType>());
  auto op_out_type = shard_op.out().type().dyn_cast<DistDenseTensorType>();
  EXPECT_EQ(op_out_type.global_ddim(), phi::make_ddim(data_shape));
  EXPECT_EQ(op_out_type.local_ddim(), phi::make_ddim(local_shape));
  EXPECT_EQ(op_out_type.process_mesh_attr(), mesh_attr);
  EXPECT_EQ(op_out_type.dims_mapping(), dims_mapping);
  EXPECT_EQ(op_out_type.partial_dims().size(), (size_t)0);

  EXPECT_EQ(
      shard_op.attribute<OperationDistAttribute>("op_dist_attr").num_operands(),
      (uint32_t)0);
  EXPECT_EQ(
      shard_op.attribute<OperationDistAttribute>("op_dist_attr").num_results(),
      (uint32_t)1);
  EXPECT_EQ(shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            mesh_attr);

  // check reshard
  std::vector<int64_t> dst_mesh_shape = {3, 2};
  phi::distributed::ProcessMesh dst_process_mesh(
      dst_mesh_shape, process_ids, dim_names);
  auto dst_mesh_attr = ProcessMeshAttribute::get(ctx, dst_process_mesh);
  auto dst_tensor_dist_attr = TensorDistAttribute::get(
      ctx, dst_mesh_attr, dims_mapping, partial_status);
  paddle::dialect::ReshardOp reshard_op =
      builder.Build<paddle::dialect::ReshardOp>(shard_op.out(),
                                                dst_tensor_dist_attr);

  EXPECT_TRUE(reshard_op.result(0).type().isa<DistDenseTensorType>());
  auto dst_op_out_type =
      reshard_op.result(0).type().dyn_cast<DistDenseTensorType>();
  EXPECT_EQ(dst_op_out_type.global_ddim(), phi::make_ddim(data_shape));
  EXPECT_EQ(dst_op_out_type.local_ddim(), phi::make_ddim({6, 6}));
  EXPECT_EQ(dst_op_out_type.process_mesh_attr(), dst_mesh_attr);
  EXPECT_EQ(dst_op_out_type.dims_mapping(), dims_mapping);
  EXPECT_EQ(dst_op_out_type.partial_dims().size(), (size_t)0);

  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_operands(),
            (uint32_t)1);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_results(),
            (uint32_t)1);
  phi::distributed::ProcessMesh flatten_process_mesh(
      {6}, process_ids, {"merged"});
  auto flatten_mesh_attr = ProcessMeshAttribute::get(ctx, flatten_process_mesh);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            flatten_mesh_attr);
}

TEST(shard_tensor_op_shard_col_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  auto mesh_attr = ProcessMeshAttribute::get(ctx, process_mesh);

  std::vector<int64_t> data_shape = {12, 6};
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status;

  // construct a col shard
  std::vector<int64_t> dims_mapping = {-1, 0};

  auto data_op = builder.Build<paddle::dialect::DataOp>(
      "w2", data_shape, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<int64_t> local_shape = {12, 3};
  auto tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, dims_mapping, partial_status);

  pir::AttributeMap attr_map = {{"tensor_dist_attr", tensor_dist_attr}};
  paddle::dialect::ShardTensorOp shard_op =
      builder.Build<paddle::dialect::ShardTensorOp>(data_op.result(0),
                                                    attr_map);

  EXPECT_TRUE(shard_op.out().type().isa<DistDenseTensorType>());
  auto op_out_type = shard_op.out().type().dyn_cast<DistDenseTensorType>();
  EXPECT_EQ(op_out_type.global_ddim(), phi::make_ddim(data_shape));
  EXPECT_EQ(op_out_type.local_ddim(), phi::make_ddim(local_shape));
  EXPECT_EQ(op_out_type.process_mesh_attr(), mesh_attr);
  EXPECT_EQ(op_out_type.dims_mapping(), dims_mapping);
  EXPECT_EQ(op_out_type.partial_dims().size(), (size_t)0);

  EXPECT_EQ(
      shard_op.attribute<OperationDistAttribute>("op_dist_attr").num_operands(),
      (uint32_t)0);
  EXPECT_EQ(
      shard_op.attribute<OperationDistAttribute>("op_dist_attr").num_results(),
      (uint32_t)1);
  EXPECT_EQ(shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            mesh_attr);

  // check reshard
  std::vector<int64_t> dst_dims_mapping = {0, 1};
  phi::distributed::ProcessMesh dst_process_mesh(
      mesh_shape, process_ids, dim_names);
  auto dst_mesh_attr = ProcessMeshAttribute::get(ctx, dst_process_mesh);
  auto dst_tensor_dist_attr = TensorDistAttribute::get(
      ctx, dst_mesh_attr, dst_dims_mapping, partial_status);
  paddle::dialect::ReshardOp reshard_op =
      builder.Build<paddle::dialect::ReshardOp>(shard_op.out(),
                                                dst_tensor_dist_attr);

  EXPECT_TRUE(reshard_op.result(0).type().isa<DistDenseTensorType>());
  auto dst_op_out_type =
      reshard_op.result(0).type().dyn_cast<DistDenseTensorType>();
  EXPECT_EQ(dst_op_out_type.global_ddim(), phi::make_ddim(data_shape));
  EXPECT_EQ(dst_op_out_type.local_ddim(), phi::make_ddim({6, 2}));
  EXPECT_EQ(dst_op_out_type.process_mesh_attr(), dst_mesh_attr);
  EXPECT_EQ(dst_op_out_type.dims_mapping(), dst_dims_mapping);
  EXPECT_EQ(dst_op_out_type.partial_dims().size(), (size_t)0);

  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_operands(),
            (uint32_t)1);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_results(),
            (uint32_t)1);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            mesh_attr);
}

TEST(mix_to_dist_pass_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  auto mesh_attr = ProcessMeshAttribute::get(ctx, process_mesh);
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status;
  std::vector<int64_t> x_shape = {12, 6};
  std::vector<int64_t> y_shape = {6, 8};

  // construct x
  std::vector<int64_t> x_dims_mapping = {0, 1};
  auto x_data_op = builder.Build<paddle::dialect::DataOp>(
      "x", x_shape, phi::DataType::FLOAT32, phi::CPUPlace());
  std::vector<int64_t> x_local_shape = {6, 2};
  auto x_tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, x_dims_mapping, partial_status);
  pir::AttributeMap x_attr_map = {{"tensor_dist_attr", x_tensor_dist_attr}};

  // construct y
  std::vector<int64_t> y_dims_mapping = {1, -1};
  auto y_data_op = builder.Build<paddle::dialect::DataOp>(
      "y", y_shape, phi::DataType::FLOAT32, phi::CPUPlace());
  std::vector<int64_t> y_local_shape = {2, 8};
  auto y_tensor_dist_attr =
      TensorDistAttribute::get(ctx, mesh_attr, y_dims_mapping, partial_status);
  pir::AttributeMap y_attr_map = {{"tensor_dist_attr", y_tensor_dist_attr}};

  // shard_tensor op
  paddle::dialect::ShardTensorOp x_shard_op =
      builder.Build<paddle::dialect::ShardTensorOp>(x_data_op.result(0),
                                                    x_attr_map);
  paddle::dialect::ShardTensorOp y_shard_op =
      builder.Build<paddle::dialect::ShardTensorOp>(y_data_op.result(0),
                                                    y_attr_map);
  EXPECT_EQ(x_shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_results(),
            (uint32_t)1);
  EXPECT_EQ(y_shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_results(),
            (uint32_t)1);
}
