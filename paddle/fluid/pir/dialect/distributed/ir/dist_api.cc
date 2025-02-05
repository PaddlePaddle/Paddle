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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_api.h"
#include <vector>
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/reduce_type.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/utils/flat_hash_map.h"

namespace paddle::dialect {

pir::Value shard_tensor(
    const pir::Value& x,
    const phi::distributed::ProcessMesh& process_mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  // support amp for shard_tensor in the future
  pir::AttributeMap attribute_map = {
      {"tensor_dist_attr",
       TensorDistAttribute::get(
           ctx, process_mesh, dims_mapping, partial_status)}};

  auto shard_tensor_op =
      ApiBuilder::Instance().GetBuilder()->Build<ShardTensorOp>(x,
                                                                attribute_map);
  return shard_tensor_op.out();
}

pir::Value reshard(
    const pir::Value& x,
    const phi::distributed::ProcessMesh& process_mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status,
    const phi::distributed::Placements& placements) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  PlacementsAttribute placements_attr =
      PlacementsAttribute::get(ctx, placements);
  TensorDistAttribute tensor_dist_attr = TensorDistAttribute::get(
      ctx, process_mesh, dims_mapping, partial_status, placements_attr);
  return reshard(x, tensor_dist_attr);
}

pir::Value reshard(const pir::Value& x,
                   const TensorDistAttribute& tensor_dist_attr) {
  auto reshard_op = ApiBuilder::Instance().GetBuilder()->Build<ReshardOp>(
      x, tensor_dist_attr);
  return reshard_op.result(0);
}

pir::Value dtensor_from_local(
    const pir::Value& x,
    const phi::distributed::ProcessMesh& process_mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  TensorDistAttribute tensor_dist_attr =
      TensorDistAttribute::get(ctx, process_mesh, dims_mapping, partial_status);
  return dtensor_from_local(x, tensor_dist_attr);
}

pir::Value dtensor_from_local(const pir::Value& x,
                              const TensorDistAttribute& tensor_dist_attr) {
  return ApiBuilder::Instance()
      .GetBuilder()
      ->Build<DtensorFromLocalOp>(x, tensor_dist_attr)
      .result(0);
}

pir::Value dtensor_to_local(const pir::Value& x) {
  return ApiBuilder::Instance().GetBuilder()->Build<DtensorToLocalOp>(x).result(
      0);
}

std::vector<pir::Value> moe_sub_mesh_tensors(
    const pir::Value& input,
    const std::vector<phi::distributed::ProcessMesh>& local_mesh_list,
    const std::vector<int64_t>& local_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& local_partial_status,
    const phi::distributed::ProcessMesh& global_mesh,
    const std::vector<int64_t>& global_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& global_partial_status) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::vector<TensorDistAttribute> local_dist_attrs;
  for (const phi::distributed::ProcessMesh& mesh : local_mesh_list) {
    local_dist_attrs.emplace_back(TensorDistAttribute::get(
        ctx, mesh, local_dims_mapping, local_partial_status));
  }
  TensorDistAttribute global_dist_attr = TensorDistAttribute::get(
      ctx, global_mesh, global_dims_mapping, global_partial_status);

  auto op = ApiBuilder::Instance().GetBuilder()->Build<MoESubMeshTensorsOp>(
      input, local_dist_attrs, global_dist_attr);
  return op.results();
}

pir::Value moe_global_mesh_tensor(
    const std::vector<pir::Value>& inputs,
    const std::vector<phi::distributed::ProcessMesh>& local_mesh_list,
    const std::vector<int64_t>& local_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& local_partial_status,
    const phi::distributed::ProcessMesh& global_mesh,
    const std::vector<int64_t>& global_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& global_partial_status,
    const std::vector<int64_t>& global_shape) {
  pir::IrContext* ctx = pir::IrContext::Instance();

  std::vector<TensorDistAttribute> local_dist_attrs;
  for (const phi::distributed::ProcessMesh& mesh : local_mesh_list) {
    local_dist_attrs.emplace_back(TensorDistAttribute::get(
        ctx, mesh, local_dims_mapping, local_partial_status));
  }

  TensorDistAttribute global_dist_attr = TensorDistAttribute::get(
      ctx, global_mesh, global_dims_mapping, global_partial_status);

  phi::DDim global_ddim = phi::make_ddim(global_shape);

  auto op = ApiBuilder::Instance().GetBuilder()->Build<MoEGlobalMeshTensorOp>(
      inputs, local_dist_attrs, global_dist_attr, global_ddim);
  return op.result(0);
}

pir::Value dist_reshape(
    const pir::Value& x,
    const phi::distributed::Placements& x_placements,
    const std::vector<int64_t>& global_shape,
    const std::vector<int64_t>& local_shape,
    const phi::distributed::ProcessMesh& mesh,
    const phi::distributed::Placements& placements,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  common::DDim global_dims = common::make_ddim(global_shape);
  common::DDim local_dims = common::make_ddim(local_shape);
  PlacementsAttribute x_placements_attr =
      PlacementsAttribute::get(ctx, x_placements);
  PlacementsAttribute placements_attr =
      PlacementsAttribute::get(ctx, placements);
  TensorDistAttribute out_dist_attr = TensorDistAttribute::get(
      ctx, mesh, dims_mapping, partial_status, placements_attr);
  auto op = ApiBuilder::Instance().GetBuilder()->Build<DistReshapeOp>(
      x, x_placements_attr, global_dims, local_dims, out_dist_attr);
  return op.result(0);
}

}  // namespace paddle::dialect
