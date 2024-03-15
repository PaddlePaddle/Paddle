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

namespace paddle {
namespace dialect {

bool HasDistInput(const std::vector<pir::Value>& inputs) {
  for (auto value : inputs) {
    if (value.type().isa<DistDenseTensorType>()) {
      return true;
    }
  }
  return false;
}

bool AllInputAreDist(const std::vector<pir::Value>& inputs) {
  for (auto value : inputs) {
    if (!value.type().isa<DistDenseTensorType>()) {
      return false;
    }
  }
  return true;
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
  return TensorDistAttribute::get(pir::IrContext::Instance(),
                                  attr.process_mesh(),
                                  attr.dims_mapping(),
                                  attr.partial_status());
}

}  // namespace dialect
}  // namespace paddle
