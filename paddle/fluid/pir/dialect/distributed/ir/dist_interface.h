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
#pragma once

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/pir/include/core/cast_utils.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/type.h"

namespace paddle {
namespace dialect {

class IR_API DistTypeInterface
    : public pir::TypeInterfaceBase<DistTypeInterface> {
 public:
  struct Concept {
    /// Defined these methods with the interface.
    explicit Concept(
        pir::Type (*local_type)(pir::Type),
        ProcessMeshAttribute (*process_mesh_attr)(pir::Type),
        TensorDistAttribute (*tensor_dist_attr)(pir::Type),
        pir::Type (*copy_with_new_mesh)(pir::Type, ProcessMeshAttribute mesh),
        pir::Type (*copy_with_new_dist_attr)(pir::Type,
                                             TensorDistAttribute dist_attr))
        : local_type(local_type),
          process_mesh_attr(process_mesh_attr),
          tensor_dist_attr(tensor_dist_attr),
          copy_with_new_mesh(copy_with_new_mesh),
          copy_with_new_dist_attr(copy_with_new_dist_attr) {}
    pir::Type (*local_type)(pir::Type);
    ProcessMeshAttribute (*process_mesh_attr)(pir::Type);
    TensorDistAttribute (*tensor_dist_attr)(pir::Type);
    pir::Type (*copy_with_new_mesh)(pir::Type, ProcessMeshAttribute mesh);
    pir::Type (*copy_with_new_dist_attr)(pir::Type,
                                         TensorDistAttribute dist_attr);
  };

  template <class ConcreteType>
  struct Model : public Concept {
    static Type local_type(Type type) {
      return pir::cast<ConcreteType>(type).local_type();
    }
    static ProcessMeshAttribute process_mesh_attr(Type type) {
      return pir::cast<ConcreteType>(type).process_mesh_attr();
    }

    static TensorDistAttribute tensor_dist_attr(Type type) {
      return pir::cast<ConcreteType>(type).tensor_dist_attr();
    }

    static Type CopyWithNewMesh(Type type, ProcessMeshAttribute mesh) {
      return pir::cast<ConcreteType>(type).CopyWithNewMesh(mesh);
    }

    static Type CopyWithNewDistAttr(Type type, TensorDistAttribute dist_attr) {
      return pir::cast<ConcreteType>(type).CopyWithNewDistAttr(dist_attr);
    }

    Model()
        : Concept(local_type,
                  process_mesh_attr,
                  tensor_dist_attr,
                  CopyWithNewMesh,
                  CopyWithNewDistAttr) {}
  };

  DistTypeInterface(pir::Type type, Concept *impl)
      : pir::TypeInterfaceBase<DistTypeInterface>(type), impl_(impl) {}

  pir::Type local_type() { return impl_->local_type(*this); }

  ProcessMeshAttribute process_mesh_attr() {
    return impl_->process_mesh_attr(*this);
  }

  TensorDistAttribute tensor_dist_attr() {
    return impl_->tensor_dist_attr(*this);
  }

  DistTypeInterface CopyWithNewMesh(ProcessMeshAttribute mesh) {
    return DistTypeInterface(impl_->copy_with_new_mesh(*this, mesh), impl_);
  }

  DistTypeInterface CopyWithNewDistAttr(TensorDistAttribute dist_attr) {
    return DistTypeInterface(impl_->copy_with_new_dist_attr(*this, dist_attr),
                             impl_);
  }

 private:
  Concept *impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DistTypeInterface)
