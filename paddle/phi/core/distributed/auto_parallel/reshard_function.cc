// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

namespace phi {
namespace distributed {

ReshardFunction* ChooseProperReshardFunction(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  for (const auto& func : GetReshardFunctionList()) {
    if (func->IsSuitable(in, out_dist_attr)) {
      return func.get();
    }
  }
  PADDLE_THROW(phi::errors::Unimplemented(
      "Can not reshard from in_dist_attr=%s to out_dist_attr=%s.",
      in.dist_attr().to_string(),
      out_dist_attr.to_string()));
}

std::vector<std::unique_ptr<ReshardFunction>>& GetReshardFunctionList() {
  static std::vector<std::unique_ptr<ReshardFunction>> func_list;
  return func_list;
}

}  // namespace distributed
}  // namespace phi
