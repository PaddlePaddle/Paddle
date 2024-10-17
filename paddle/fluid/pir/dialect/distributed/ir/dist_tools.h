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
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace dialect {

ProcessMeshAttribute MergeMeshes(const ProcessMeshAttribute& mesh1,
                                 const ProcessMeshAttribute& mesh2);

ProcessMeshAttribute MergeInputMeshes(const std::vector<pir::Value>& inputs);

ProcessMeshAttribute CreateGlobalMesh(const std::vector<pir::Value>& inputs);

bool HasDistInput(const std::vector<pir::Value>& inputs,
                  ProcessMeshAttribute* p_mesh_attr = nullptr);
bool AllInputAreDist(const std::vector<pir::Value>& inputs);

pir::Attribute GetTensorDistAttr(pir::Type type);

void CvtAllInputsToDist(const std::vector<pir::Value>& inputs,
                        ProcessMeshAttribute mesh_attr);

phi::distributed::DistMetaTensor CvtToDistMetaTensor(DistDenseTensorType type);

std::vector<phi::distributed::DistMetaTensor> CvtToDistMetaTensor(
    pir::VectorType type);
pir::Attribute CvtToPirAttr(const phi::distributed::ArgDistAttr& dist_attr);

pir::Attribute CreateReplicatedDistAttr(pir::Type prim_type,
                                        ProcessMeshAttribute mesh);

pir::Type CvtToPirDistType(
    pir::Type global_type,
    pir::Attribute dist_attr,
    const std::vector<int64_t>& local_ddim = std::vector<int64_t>());

///
/// When the following conditions are met:
///    1. The value's type is dist type.
///    2. The value type's mesh is not equal to mesh_attr argument.
///    3. The operation that defines the value contains no inputs and 1 output.
/// The function first clones the definition operation and replaces the use of
/// the original value with the cloned output， Secondly, the mesh of the
/// original operation and value is updated with the 'mesh_attr' argument.
/// Otherwise, the function does nothing.
///
void CopyLeafOpToMesh(pir::Value value, ProcessMeshAttribute mesh_attr);

}  // namespace dialect
}  // namespace paddle
