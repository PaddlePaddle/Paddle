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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"

#include "paddle/fluid/pir/dialect/distributed/ir/attribute_storage.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/distributed/ir/type_storage.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

REGISTER_FILE_SYMBOLS(dist_dialect);
namespace paddle {
namespace dialect {

DistDialect::DistDialect(pir::IrContext *context)
    : pir::Dialect(name(), context, pir::TypeId::get<DistDialect>()) {
  initialize();
}

void DistDialect::initialize() {
  RegisterAttributes<ProcessMeshAttribute,
                     TensorDistAttribute,
                     OperationDistAttribute>();
  RegisterTypes<DistDenseTensorType>();
  RegisterOps<ShardTensorOp>();
}

void DistDialect::PrintType(pir::Type type, std::ostream &os) const {
  if (auto dist_dense_tensor_type = type.dyn_cast<DistDenseTensorType>()) {
    // Todo: Design the dist dense tensor type print format.
    os << type.dialect().name();
    os << '.';
    if (auto tensor_type = type.dyn_cast<pir::DenseTensorType>()) {
      os << "tensor<";
      for (auto d : common::vectorize(tensor_type.dims())) {
        os << d;
        os << "x";
      }
      tensor_type.dtype().Print(os);
      os << ", ";
      PrintAttribute(dist_dense_tensor_type.tensor_dist_attr(), os);
      os << ">";
    }
  } else {
    os << "error_type!";
  }
}

void DistDialect::PrintAttribute(pir::Attribute attr, std::ostream &os) const {
  if (auto process_mesh_attr = attr.dyn_cast<ProcessMeshAttribute>()) {
    os << "mesh_shape:[" +
              phi::distributed::auto_parallel::str_join(
                  process_mesh_attr.shape()) +
              "]";
    os << ",process_ids:[" +
              phi::distributed::auto_parallel::str_join(
                  process_mesh_attr.process_ids()) +
              "]";
  } else if (auto tensor_dist_attr = attr.dyn_cast<TensorDistAttribute>()) {
    // Todo: Design the tensor dist attr print format.
    os << "mesh_shape:[" +
              phi::distributed::auto_parallel::str_join(
                  tensor_dist_attr.process_mesh_attr().shape()) +
              "]";
    os << ",dims_mappings:[" +
              phi::distributed::auto_parallel::str_join(
                  tensor_dist_attr.dims_mapping()) +
              "]";
    if (tensor_dist_attr.partial_status().size() > 0) {
      std::vector<std::string> partial_status_strs;
      for (auto &itr : tensor_dist_attr.partial_status()) {
        std::string s = "partial(" + std::to_string(itr.first) + "," +
                        phi::ReduceTypeStrings[static_cast<int>(itr.second)] +
                        ")";
        partial_status_strs.emplace_back(s);
      }
      os << ", "
         << phi::distributed::auto_parallel::str_join(partial_status_strs);
    }
  } else if (auto op_dist_attr = attr.dyn_cast<OperationDistAttribute>()) {
    os << "mesh_shape:[" +
              phi::distributed::auto_parallel::str_join(
                  op_dist_attr.process_mesh_attr().shape()) +
              "]";
    os << ",process_ids:[" +
              phi::distributed::auto_parallel::str_join(
                  op_dist_attr.process_mesh_attr().process_ids()) +
              "]";
    auto num_operand_dist_attrs = op_dist_attr.num_operand_dist_attrs();
    for (uint32_t i = 0; i < num_operand_dist_attrs; ++i) {
      auto dist_attr = op_dist_attr.operand_dist_attr(i);
      os << ",operand(" + std::to_string(i) + "):{";
      if (dist_attr.process_mesh_attr() != op_dist_attr.process_mesh_attr()) {
        os << "mesh_shape:[" +
                  phi::distributed::auto_parallel::str_join(
                      dist_attr.process_mesh_attr().shape()) +
                  "],";
      }
      os << "dims_maping:[" +
                phi::distributed::auto_parallel::str_join(
                    dist_attr.dims_mapping()) +
                "]";
      if (dist_attr.partial_status().size() > 0) {
        std::vector<std::string> partial_status_strs;
        for (auto &itr : dist_attr.partial_status()) {
          std::string s = "partial(" + std::to_string(itr.first) + "," +
                          phi::ReduceTypeStrings[static_cast<int>(itr.second)] +
                          ")";
          partial_status_strs.emplace_back(s);
        }
        os << "," +
                  phi::distributed::auto_parallel::str_join(
                      partial_status_strs) +
                  "}";
      } else {
        os << "}";
      }
    }
    auto num_result_dist_attrs = op_dist_attr.num_result_dist_attrs();
    for (uint32_t i = 0; i < num_result_dist_attrs; ++i) {
      auto dist_attr = op_dist_attr.result_dist_attr(i);
      os << ",result(" + std::to_string(i) + "):{";
      if (dist_attr.process_mesh_attr() != op_dist_attr.process_mesh_attr()) {
        os << "mesh_shape:[" +
                  phi::distributed::auto_parallel::str_join(
                      dist_attr.process_mesh_attr().shape()) +
                  "],";
      }
      os << "dims_maping:[" +
                phi::distributed::auto_parallel::str_join(
                    dist_attr.dims_mapping()) +
                "]";
      if (dist_attr.partial_status().size() > 0) {
        std::vector<std::string> partial_status_strs;
        for (auto &itr : dist_attr.partial_status()) {
          std::string s = "partial(" + std::to_string(itr.first) + "," +
                          phi::ReduceTypeStrings[static_cast<int>(itr.second)] +
                          ")";
          partial_status_strs.emplace_back(s);
        }
        os << "," +
                  phi::distributed::auto_parallel::str_join(
                      partial_status_strs) +
                  "}";
      } else {
        os << "}";
      }
    }
  } else {
    os << "error_attribute_type";
  }
}

pir::OpPrintFn DistDialect::PrintOperation(pir::Operation *op) const {
  return nullptr;
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DistDialect)
