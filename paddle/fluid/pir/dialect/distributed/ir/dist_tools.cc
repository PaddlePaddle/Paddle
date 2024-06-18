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
#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/operation.h"

namespace paddle::dialect {

bool AllInputAreDist(const std::vector<pir::Value>& inputs) {
  for (auto value : inputs) {
    auto type = value.type();
    if (!type || type.isa<DistTypeInterface>()) continue;
    if (auto vec_type = value.type().dyn_cast<pir::VectorType>()) {
      for (size_t idx = 0; idx < vec_type.size(); ++idx) {
        if (vec_type[idx] && !vec_type[idx].isa<DistTypeInterface>()) {
          return false;
        }
      }
    } else {
      return false;
    }
  }
  return true;
}

bool HasDistInput(const std::vector<pir::Value>& inputs,
                  ProcessMeshAttribute* p_mesh_attr) {
  for (auto value : inputs) {
    if (auto dist_type = value.type().dyn_cast<DistTypeInterface>()) {
      if (p_mesh_attr) {
        *p_mesh_attr = dist_type.process_mesh_attr();
      }
      return true;
    } else {
      auto vec_type = value.type().dyn_cast<pir::VectorType>();
      if (!vec_type) {
        continue;
      }
      for (size_t idx = 0; idx < vec_type.size(); ++idx) {
        if (auto dist_type = vec_type[idx].dyn_cast<DistTypeInterface>()) {
          if (p_mesh_attr) {
            *p_mesh_attr = dist_type.process_mesh_attr();
          }
          return true;
        }
      }
      return false;
    }
  }
  return false;
}

void CvtAllInputsToDist(const std::vector<pir::Value>& inputs,
                        ProcessMeshAttribute mesh_attr) {
  for (auto value : inputs) {
    if (auto type = value.type()) {
      if (type.isa<DistTypeInterface>() || type.isa<pir::VectorType>())
        continue;
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

std::vector<phi::distributed::DistMetaTensor> CvtToDistMetaTensor(
    pir::VectorType type) {
  if (!type) return {};
  std::vector<phi::distributed::DistMetaTensor> res;
  for (size_t idx = 0; idx < type.size(); ++idx) {
    if (auto dist_type = type[idx].dyn_cast<DistDenseTensorType>()) {
      res.emplace_back(CvtToDistMetaTensor(dist_type));
    } else {
      res.emplace_back(phi::distributed::DistMetaTensor());
    }
  }
  return res;
}

pir::Attribute CvtToPirAttr(const phi::distributed::ArgDistAttr& dist_attr) {
  auto ctx = pir::IrContext::Instance();
  if (holds_alternative<phi::distributed::TensorDistAttr>(dist_attr)) {
    auto& attr = PADDLE_GET_CONST(phi::distributed::TensorDistAttr, dist_attr);
    return TensorDistAttribute::get(
        ctx, attr.process_mesh(), attr.dims_mapping(), attr.partial_status());
  } else {
    auto& vec = PADDLE_GET_CONST(std::vector<phi::distributed::TensorDistAttr>,
                                 dist_attr);
    std::vector<pir::Attribute> array;
    array.reserve(vec.size());
    for (auto& attr : vec) {
      array.push_back(TensorDistAttribute::get(ctx,
                                               attr.process_mesh(),
                                               attr.dims_mapping(),
                                               attr.partial_status()));
    }
    return pir::ArrayAttribute::get(ctx, array);
  }
}

pir::Attribute CreateReplicatedDistAttr(pir::Type prim_type,
                                        ProcessMeshAttribute mesh) {
  auto ctx = pir::IrContext::Instance();
  if (auto tensor_type = prim_type.dyn_cast<pir::DenseTensorType>()) {
    auto& ddim = tensor_type.dims();
    return TensorDistAttribute::get(
        ctx, mesh, std::vector<int64_t>(ddim.size(), -1));
  } else if (auto vec_type = prim_type.dyn_cast<pir::VectorType>()) {
    std::vector<pir::Attribute> array;
    for (size_t idx = 0; idx < vec_type.size(); ++idx) {
      array.emplace_back(CreateReplicatedDistAttr(vec_type[idx], mesh));
    }
    return pir::ArrayAttribute::get(ctx, array);
  }
  return nullptr;
}
pir::Type CvtToPirDistType(pir::Type global_type, pir::Attribute dist_attr) {
  if (!global_type) return nullptr;
  auto ctx = pir::IrContext::Instance();
  if (auto dense_tensor_type = global_type.dyn_cast<pir::DenseTensorType>()) {
    auto tensor_dist_attr = dist_attr.dyn_cast<TensorDistAttribute>();
    if (!tensor_dist_attr) {
      VLOG(0) << "Convert dense tensor type to dist type with attribute {"
              << dist_attr << "}.";
      PADDLE_THROW(common::errors::InvalidArgument(
          "Only allowed convert a densor tensor type to dist dense tensor type "
          "with non-empty TensorDistAttr"));
    }
    return DistDenseTensorType::get(ctx, dense_tensor_type, tensor_dist_attr);
  } else if (auto vec_type = global_type.dyn_cast<pir::VectorType>()) {
    auto array_attr = dist_attr.dyn_cast<pir::ArrayAttribute>();
    if (!array_attr) {
      VLOG(0) << "Convert vector type to dist type with attribute {"
              << dist_attr << "}.";
      PADDLE_THROW(common::errors::InvalidArgument(
          "Only allowed convert a vector type to dist vector type with "
          "non-empty ArrayAttr"));
    }
    PADDLE_ENFORCE_EQ(
        vec_type.size(),
        array_attr.size(),
        common::errors::InvalidArgument(
            "The vector type size must equal to array attribute size."));
    std::vector<pir::Type> dist_vec_type;
    for (size_t idx = 0; idx < vec_type.size(); ++idx) {
      dist_vec_type.push_back(CvtToPirDistType(vec_type[idx], array_attr[idx]));
    }
    return pir::VectorType::get(ctx, dist_vec_type);
  } else {
    VLOG(0) << "Convert type{" << global_type
            << "} to dist type with attribute {" << dist_attr << "}.";
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently only support convert dense_tensor_type r vector type to "
        "dist."));
  }
  return nullptr;
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
}  // namespace paddle::dialect
