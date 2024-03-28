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
#include "paddle/fluid/pir/dialect/distributed/transforms/mix_to_dist_pass.h"
#include "paddle/fluid/pir/dialect/distributed/transforms/reshard_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/program.h"

using namespace paddle::dialect;  // NOLINT

TEST(shard_tensor_op_replicate_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<DistDialect>();
  ctx->GetOrRegisterDialect<OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  std::vector<int64_t> mesh_shape = {2};
  std::vector<int64_t> process_ids = {0, 1};
  std::vector<std::string> dim_names = {"x"};
  phi::distributed::ProcessMesh process_mesh(
      mesh_shape, process_ids, dim_names);
  auto mesh_attr = ProcessMeshAttribute::get(ctx, process_mesh);

  std::vector<int64_t> data_shape = {12, 6};
  paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status{{0, phi::ReduceType::kRedSum}};

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
  EXPECT_EQ(op_out_type.partial_dims().size(), (size_t)1);

  EXPECT_EQ(shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_operand_dist_attrs(),
            (uint32_t)0);

  EXPECT_EQ(shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_result_dist_attrs(),
            (uint32_t)1);
  EXPECT_EQ(shard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            mesh_attr);

  // check reshard
  std::vector<int64_t> dst_mesh_shape = {2};
  std::vector<int64_t> dst_dims_mapping = {-1, -1};

  phi::distributed::ProcessMesh dst_process_mesh(
      dst_mesh_shape, process_ids, dim_names);
  auto dst_mesh_attr = ProcessMeshAttribute::get(ctx, dst_process_mesh);

  paddle::flat_hash_map<int64_t, phi::ReduceType> dst_partial_status;
  auto dst_tensor_dist_attr = TensorDistAttribute::get(
      ctx, dst_mesh_attr, dst_dims_mapping, dst_partial_status);
  paddle::dialect::ReShardOp reshard_op =
      builder.Build<paddle::dialect::ReShardOp>(shard_op.out(),
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
                .num_operand_dist_attrs(),
            (uint32_t)1);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .num_result_dist_attrs(),
            (uint32_t)1);
  EXPECT_EQ(reshard_op.attribute<OperationDistAttribute>("op_dist_attr")
                .process_mesh_attr(),
            mesh_attr);
  
  // check reshard_pass
  std::cout << "IR before Reshard Pass = " << program << std::endl;
  std::shared_ptr<pir::Program> new_program = paddle::dialect::ReshardPass(&program);
  std::cout << "IR after Reshard Pass = " << new_program << std::endl;
  //pir::Block* new_block = new_program->block();
}

