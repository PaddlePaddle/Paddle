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
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/distributed/ir/type_storage.h"

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
}

void DistDialect::PrintType(pir::Type type, std::ostream &os) const {
  if (auto dist_dense_tensor_type = type.dyn_cast<DistDenseTensorType>()) {
    // Todo: Design the dist dense tensor type print format.
    os << dist_dense_tensor_type.dense_tensor_type();
  } else {
    os << "error_type!";
  }
}

void DistDialect::PrintAttribute(pir::Attribute attr, std::ostream &os) const {
  if (auto process_mesh_attr = attr.dyn_cast<ProcessMeshAttribute>()) {
    os << process_mesh_attr.process_mesh();
  } else if (auto tensor_dist_attr = attr.dyn_cast<TensorDistAttribute>()) {
    // Todo: Design the tensor dist attr print format.
    os << tensor_dist_attr.process_mesh_attr().process_mesh();
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
