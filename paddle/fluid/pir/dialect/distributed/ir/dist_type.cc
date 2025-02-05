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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/distributed/ir/type_storage.h"
#include "paddle/pir/include/core/ir_context.h"

namespace paddle::dialect {

pir::DenseTensorType DistDenseTensorType::dense_tensor_type() const {
  return storage()->dense_tensor_type;
}

TensorDistAttribute DistDenseTensorType::tensor_dist_attr() const {
  return storage()->tensor_dist_attr;
}

const common::DDim& DistDenseTensorType::local_ddim() const {
  return storage()->local_ddim;
}

DistDenseTensorType DistDenseTensorType::get(
    pir::IrContext* ctx,
    pir::DenseTensorType dense_tensor_type,
    TensorDistAttribute tensor_dist_attr,
    const common::DDim& local_ddim) {
  return Base::get(ctx, dense_tensor_type, tensor_dist_attr, local_ddim);
}

common::DDim InferLocalDDim(const common::DDim& global_ddim,
                            TensorDistAttribute dist_attr) {
  if (global_ddim.size() == -1 || global_ddim.size() == 0) {
    return global_ddim;
  }
  const ProcessMeshAttribute& mesh_attr = dist_attr.process_mesh_attr();
  auto& mesh_dim = mesh_attr.shape();
  auto& dim_mapping = dist_attr.dims_mapping();
  PADDLE_ENFORCE_EQ(global_ddim.size(),
                    dim_mapping.size(),
                    ::common::errors::PreconditionNotMet(
                        "The global ddim size must equal to dim_mapping's "
                        "size, but bot %d vs %d",
                        global_ddim.size(),
                        dim_mapping.size()));

  common::DDim local_ddim(global_ddim);
  if (dist_attr.placements_attr().has_value()) {
    PlacementsAttribute placements_attr = dist_attr.placements_attr().value();
    const phi::distributed::Placements& placements =
        placements_attr.placements();
    for (size_t i = 0; i < placements.size(); i++) {
      if (placements[i]->is_shard()) {
        int tensor_dim =
            dynamic_cast<const phi::distributed::Shard&>(*placements[i])
                .get_dim();
        auto dim_size = mesh_dim.at(i);
        local_ddim[tensor_dim] =
            (local_ddim[tensor_dim] + dim_size - 1) / dim_size;
      }
    }
  } else {
    for (size_t i = 0; i < dim_mapping.size(); ++i) {
      if (dim_mapping[i] != -1) {
        auto dim_size = mesh_dim.at(dim_mapping[i]);
        local_ddim[i] = (global_ddim[i] + dim_size - 1) / dim_size;
      }
    }
  }
  return local_ddim;
}

common::DDim InferGlobalDDim(const common::DDim& local_ddim,
                             TensorDistAttribute dist_attr) {
  if (local_ddim.size() == -1 || local_ddim.size() == 0) {
    return local_ddim;
  }
  const ProcessMeshAttribute& mesh_attr = dist_attr.process_mesh_attr();
  auto& mesh_dim = mesh_attr.shape();
  auto& dim_mapping = dist_attr.dims_mapping();
  PADDLE_ENFORCE_EQ(local_ddim.size(),
                    dim_mapping.size(),
                    ::common::errors::PreconditionNotMet(
                        "The local ddim size must equal to dim_mapping's "
                        "size, but bot %d vs %d",
                        local_ddim.size(),
                        dim_mapping.size()));
  common::DDim global_ddim(local_ddim);
  for (size_t i = 0; i < dim_mapping.size(); ++i) {
    if (dim_mapping[i] != -1) {
      global_ddim[i] = local_ddim[i] * mesh_dim.at(dim_mapping[i]);
    }
  }
  return global_ddim;
}

pir::DenseTensorType DistDenseTensorType::local_type() const {
  return pir::DenseTensorType::get(pir::IrContext::Instance(),
                                   dtype(),
                                   local_ddim(),
                                   data_layout(),
                                   lod(),
                                   offset());
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DistDenseTensorType)
