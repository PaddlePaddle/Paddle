/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo EinsumInferSpmd(const std::vector<DistMetaTensor>& inputs,
                         const std::string& equation);

SpmdInfo EinsumGradInferSpmd(const std::vector<DistMetaTensor>& inputs,
                             const std::vector<DistMetaTensor>& inner_cache,
                             const DistMetaTensor& out_grad,
                             const std::string& equation);
}  // namespace distributed
}  // namespace phi
