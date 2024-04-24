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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/operation.h"

namespace paddle {
namespace dialect {

bool HasDistInput(const std::vector<pir::Value>& inputs,
                  ProcessMeshAttribute* p_mesh_attr) {
  for (auto value : inputs) {
    if (auto dist_type = value.type().dyn_cast<DistTypeInterface>()) {
      if (p_mesh_attr) {
        *p_mesh_attr = dist_type.process_mesh_attr();
      }
      return true;
    }
  }
  return false;
}

void CvtAllInputsToDist(const std::vector<pir::Value>& inputs,
                        ProcessMeshAttribute mesh_attr) {
  for (auto value : inputs) {
    if (auto type = value.type()) {
      if (type.isa<DistTypeInterface>()) continue;
      auto dense_type = type.dyn_cast<pir::DenseTensorType>();
      if (!dense_type) {
        PADDLE_THROW(common::errors::Unimplemented(
            "Currently only support convert dense_tensor_type to dist type."));
      }
      auto ctx = pir::IrContext::Instance();
      auto dist_type = DistDenseTensorType::get(ctx, dense_type, mesh_attr);
      value.set_type(dist_type);
      if (auto define_op = value.defining_op()) {
        if (define_op->num_operands() != 0u) {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Currently only allowed add dist attribue for leaf nodes "
              "operation. The current op is %s",
              define_op->name()));
        }
        if (define_op->num_results() != 1u) {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Currently only allowed add dist attribue for operation with "
              "single output. The current op is %s",
              define_op->name()));
        }
        define_op->set_attribute(
            kAttrOpDistAttr,
            OperationDistAttribute::get(
                ctx, mesh_attr, {}, {dist_type.tensor_dist_attr()}));
      }
    }
  }
}

phi::distributed::DistMetaTensor CvtToDistMetaTensor(DistDenseTensorType type) {
  auto pir_attr = type.tensor_dist_attr();
  phi::distributed::TensorDistAttr phi_attr;
  phi_attr.set_process_mesh(pir_attr.process_mesh_attr().process_mesh());
  phi_attr.set_dims_mapping(pir_attr.dims_mapping());
  phi_attr.set_partial_status(pir_attr.partial_status());
  return phi::distributed::DistMetaTensor(type.global_ddim(), phi_attr);
}

TensorDistAttribute CvtToPirDistAttr(
    const phi::distributed::ArgDistAttr& dist_attr) {
  auto& attr = PADDLE_GET_CONST(phi::distributed::TensorDistAttr, dist_attr);
  if (attr.process_mesh().empty()) return nullptr;
  return TensorDistAttribute::get(pir::IrContext::Instance(),
                                  attr.process_mesh(),
                                  attr.dims_mapping(),
                                  attr.partial_status());
}

void CopyLeafOpToMesh(pir::Value value, ProcessMeshAttribute mesh_attr) {
  if (auto dist_type = value.type().dyn_cast<DistTypeInterface>()) {
    if (dist_type.process_mesh_attr() == mesh_attr) {
      return;
    }
    if (auto op = value.defining_op()) {
      if (op->num_operands() != 0u || op->num_results() != 1u) {
        return;
      }
      pir::IrMapping ir_mapping;
      auto new_op = op->Clone(ir_mapping);
      op->GetParent()->insert(*op, new_op);
      value.ReplaceAllUsesWith(new_op->result(0));
      dist_type = dist_type.CopyWithNewMesh(mesh_attr);
      value.set_type(dist_type);
      op->set_attribute(
          kAttrOpDistAttr,
          OperationDistAttribute::get(dist_type.ir_context(),
                                      mesh_attr,
                                      {},
                                      {dist_type.tensor_dist_attr()}));
    }
  }
}
}  // namespace dialect
}  // namespace paddle
