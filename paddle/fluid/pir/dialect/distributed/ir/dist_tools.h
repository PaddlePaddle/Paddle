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

bool HasDistInput(const std::vector<pir::Value>& inputs);
bool AllInputAreDist(const std::vector<pir::Value>& inputs);
phi::distributed::DistMetaTensor CvtToDistMetaTensor(DistDenseTensorType type);
TensorDistAttribute CvtToPirDistAttr(
    const phi::distributed::ArgDistAttr& dist_attr);

}  // namespace dialect
}  // namespace paddle
