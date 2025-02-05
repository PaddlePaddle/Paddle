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

#include <unordered_set>

#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/pir/include/core/operation.h"

namespace paddle::dialect {

ProcessMeshAttribute MergeMeshes(const ProcessMeshAttribute& mesh1,
                                 const ProcessMeshAttribute& mesh2) {
  if (mesh1 == mesh2) return mesh1;
  // Combine the two ids
  std::vector<int64_t> merged_ids;
  std::vector<int64_t> ids1 = mesh1.process_ids();
  std::vector<int64_t> ids2 = mesh2.process_ids();

  merged_ids.reserve(ids1.size() + ids2.size());
  merged_ids.insert(merged_ids.end(), ids1.begin(), ids1.end());
  merged_ids.insert(merged_ids.end(), ids2.begin(), ids2.end());

  // Remove duplicates
  std::sort(merged_ids.begin(), merged_ids.end());
  auto last = std::unique(merged_ids.begin(), merged_ids.end());
  merged_ids.erase(last, merged_ids.end());

  return ProcessMeshAttribute::get(
      pir::IrContext::Instance(),
      {static_cast<int64_t>(merged_ids.size())},  // flatten mesh shape
      merged_ids,
      {"merged"});
}

ProcessMeshAttribute MergeInputMeshes(const std::vector<pir::Value>& inputs) {
  auto ctx = pir::IrContext::Instance();
  auto mesh = ProcessMeshAttribute::get(ctx, {}, {}, {});
  for (auto value : inputs) {
    if (auto dist_type = value.type().dyn_cast<DistTypeInterface>()) {
      mesh = MergeMeshes(mesh, dist_type.process_mesh_attr());
    } else {
      auto vec_type = value.type().dyn_cast<pir::VectorType>();
      if (!vec_type) {
        continue;
      }
      for (size_t idx = 0; idx < vec_type.size(); ++idx) {
        if (auto dist_type = vec_type[idx].dyn_cast<DistTypeInterface>()) {
          mesh = MergeMeshes(mesh, dist_type.process_mesh_attr());
        }
      }
    }
  }
  return mesh;
}

ProcessMeshAttribute CreateGlobalMesh(const std::vector<pir::Value>& inputs) {
  auto ctx = pir::IrContext::Instance();
  struct MyHash {
    std::size_t operator()(const ProcessMeshAttribute& obj) const {
      return obj.hash();
    }
  };
  std::unordered_set<ProcessMeshAttribute, MyHash> meshes;
  for (auto value : inputs) {
    if (auto dist_type = value.type().dyn_cast<DistTypeInterface>()) {
      meshes.insert(dist_type.process_mesh_attr());
    } else {
      if (auto vec_type = value.type().dyn_cast<pir::VectorType>()) {
        for (size_t idx = 0; idx < vec_type.size(); ++idx) {
          if (auto dist_type = vec_type[idx].dyn_cast<DistTypeInterface>()) {
            meshes.insert(dist_type.process_mesh_attr());
          }
        }
      }
    }
  }

  ProcessMeshAttribute global_mesh;
  PADDLE_ENFORCE_GT(meshes.size(),
                    0,
                    common::errors::InvalidArgument("There is no dist input"));
  // get mesh that has the most dimensions
  auto max_ndim_mesh = ProcessMeshAttribute::get(ctx, {}, {}, {});
  int64_t min_ndim = std::numeric_limits<int64_t>::max();
  for (const auto& mesh : meshes) {
    if (mesh.ndim() > max_ndim_mesh.ndim()) {
      max_ndim_mesh = mesh;
    }
    if (mesh.ndim() < min_ndim) {
      min_ndim = mesh.ndim();
    }
  }
  // min != max, means there are different mesh size
  // so, the max_ndim_mesh should be the global mesh
  if (min_ndim != max_ndim_mesh.ndim()) {
    for (const auto& mesh : meshes) {
      if (mesh != max_ndim_mesh) {
        if (!phi::distributed::IsSubMesh(max_ndim_mesh.process_mesh(),
                                         mesh.process_mesh())) {
          PADDLE_THROW(common::errors::InvalidArgument(
              "The small mesh should be the sub mesh of the large mesh, but "
              "got {%s} vs {%s} ",
              mesh,
              max_ndim_mesh));
        }
      }
    }
    global_mesh = max_ndim_mesh;
  } else {
    auto it = meshes.begin();
    auto first_mesh = *it;
    if (meshes.size() > 1) {
      auto global_ids = first_mesh.process_ids();
      auto global_shape = first_mesh.shape();
      auto global_names = first_mesh.dim_names();
      ++it;
      for (; it != meshes.end(); ++it) {
        auto mesh = *it;
        VLOG(4) << (mesh.shape() == first_mesh.shape()) << " "
                << (mesh.dim_names() == first_mesh.dim_names()) << " "
                << (mesh.process_ids() != first_mesh.process_ids());
        if (mesh.shape() == first_mesh.shape() &&
            mesh.dim_names() == first_mesh.dim_names() &&
            mesh.process_ids() != first_mesh.process_ids()) {
          global_ids.insert(global_ids.end(),
                            mesh.process_ids().begin(),
                            mesh.process_ids().end());
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "The sub meshes should have same shape and names but different "
              "process_ids, but got {%s} vs {%s} ",
              first_mesh,
              mesh));
        }
      }
      global_shape.emplace(global_shape.begin(), meshes.size());
      global_names.emplace(global_names.begin(), "global");
      global_mesh = ProcessMeshAttribute::get(
          ctx, global_shape, global_ids, global_names);
    } else {
      global_mesh = first_mesh;
    }
  }
  return global_mesh;
}

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
void GetConnectedValues(
    pir::Value value,
    std::unordered_set<pir::Value>& connected_values);  // NOLINT
void GetConnectedValues(
    pir::Operation* op,
    std::unordered_set<pir::Value>& connected_values) {  // NOLINT
  if (nullptr == op) return;
  for (auto input : op->operands_source()) {
    GetConnectedValues(input, connected_values);
  }
  for (auto output : op->results()) {
    GetConnectedValues(output, connected_values);
  }

  for (auto& block : op->blocks()) {
    for (auto arg : block.args()) {
      GetConnectedValues(arg, connected_values);
    }
  }
}

void GetConnectedValues(
    pir::Value value,
    std::unordered_set<pir::Value>& connected_values) {  // NOLINT
  if (connected_values.find(value) != connected_values.end()) return;
  connected_values.insert(value);
  GetConnectedValues(value.defining_op(), connected_values);
  for (auto iter = value.use_begin(); iter != value.use_end(); ++iter) {
    GetConnectedValues(iter->owner(), connected_values);
  }
}

// convert a single value type to dist type.
pir::Type CvtTypeToDist(pir::Type type, ProcessMeshAttribute mesh_attr) {
  if (!type) return nullptr;
  if (type.isa<DistTypeInterface>()) return type;
  auto ctx = pir::IrContext::Instance();
  if (auto dense_type = type.dyn_cast<pir::DenseTensorType>()) {
    return DistDenseTensorType::get(ctx, dense_type, mesh_attr);
  } else if (auto vec_type = type.dyn_cast<pir::VectorType>()) {
    std::vector<pir::Type> vec_dist_types;
    for (size_t idx = 0; idx < vec_type.size(); ++idx) {
      vec_dist_types.push_back(CvtTypeToDist(vec_type[idx], mesh_attr));
    }
    return pir::VectorType::get(ctx, vec_dist_types);
  }
  return type;
}

pir::Attribute GetTensorDistAttr(pir::Type type) {
  if (!type) return nullptr;
  if (auto dist_type = type.dyn_cast<DistTypeInterface>()) {
    return dist_type.tensor_dist_attr();
  } else if (auto vec_type = type.dyn_cast<pir::VectorType>()) {
    std::vector<pir::Attribute> arr_attr;
    for (size_t idx = 0; idx < vec_type.size(); ++idx) {
      arr_attr.push_back(GetTensorDistAttr(vec_type[idx]));
    }
    return pir::ArrayAttribute::get(pir::IrContext::Instance(), arr_attr);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Can't get tensor dist attribute with a non-dist type."));
  }
}

void CvtValueToDist(pir::Value value, ProcessMeshAttribute mesh_attr) {
  std::unordered_set<pir::Value> connected_values;
  GetConnectedValues(value, connected_values);
  for (auto value : connected_values) {
    value.set_type(CvtTypeToDist(value.type(), mesh_attr));
  }
}

void CvtAllInputsToDist(const std::vector<pir::Value>& inputs,
                        ProcessMeshAttribute mesh_attr) {
  for (auto value : inputs) {
    if (auto type = value.type()) {
      if (type.isa<DistTypeInterface>()) continue;
      if (auto vec_type = type.dyn_cast<pir::VectorType>()) {
        for (size_t idx = 0; idx < vec_type.size(); ++idx) {
          auto inner_type = vec_type[idx];
          if (inner_type && !inner_type.isa<DistTypeInterface>()) {
            CvtValueToDist(value, mesh_attr);
            break;
          }
        }
      } else {
        CvtValueToDist(value, mesh_attr);
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
pir::Type CvtToPirDistType(pir::Type global_type,
                           pir::Attribute dist_attr,
                           const std::vector<int64_t>& local_ddim) {
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
    if (!local_ddim.empty()) {
      return DistDenseTensorType::get(ctx,
                                      dense_tensor_type,
                                      tensor_dist_attr,
                                      common::make_ddim(local_ddim));
    } else {
      return DistDenseTensorType::get(ctx, dense_tensor_type, tensor_dist_attr);
    }
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
      dist_vec_type.push_back(
          CvtToPirDistType(vec_type[idx], array_attr[idx], local_ddim));
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
      if (mesh_attr.ndim() > 1 &&
          phi::distributed::IsSubMesh(
              mesh_attr.process_mesh(),
              dist_type.process_mesh_attr().process_mesh())) {
        auto new_dist_type = dist_type.CopyWithNewMesh(mesh_attr);
        value.set_type(new_dist_type);
        op->set_attribute(
            kAttrOpDistAttr,
            OperationDistAttribute::get(new_dist_type.ir_context(),
                                        mesh_attr,
                                        {},
                                        {new_dist_type.tensor_dist_attr()}));
        VLOG(4) << "CopyLeafOpToMesh: change mesh from "
                << dist_type.process_mesh_attr() << " to " << mesh_attr;
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
      VLOG(4) << "CopyLeafOpToMesh: copy value from "
              << dist_type.process_mesh_attr() << " to " << mesh_attr;
    }
  }
}
}  // namespace paddle::dialect
