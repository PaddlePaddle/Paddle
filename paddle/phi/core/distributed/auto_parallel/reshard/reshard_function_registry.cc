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

#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function_registry.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"

namespace phi {
namespace distributed {

ReshardFunction* ChooseProperReshardFunction(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  auto all_process_ids =
      GetUnionProcessIds(in.process_mesh().process_ids(),
                         out_dist_attr.process_mesh().process_ids());

  if (!all_process_ids.empty()) {
    auto world_size = GetGlobalWorldSize();
    auto min_value = all_process_ids.front();
    auto max_value = all_process_ids.back();
    PADDLE_ENFORCE_GE(
        min_value,
        0,
        phi::errors::OutOfRange(
            "The process id should be non-negative, but received %d. Please "
            "check the number of processes launched and process_mesh.",
            min_value));
    PADDLE_ENFORCE_LT(
        max_value,
        world_size,
        phi::errors::OutOfRange(
            "The process id should be less than %d, but received %d. Please "
            "check the number of processes launched and process_mesh.",
            world_size,
            max_value));
  }

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

// NOTE: be aware of the registration order of the reshard function.
// Higher priority will be granted to the reshard function
// which was registered earlier.
// Reshard function with higher priority will be evoked
// when more than one reshard function satisfy the request.
REGISTER_RESHARD_FUNC(SToRReshardFunction);
REGISTER_RESHARD_FUNC(SToRReshardFunctionCrossMesh);
REGISTER_RESHARD_FUNC(SToPReshardFunction);
REGISTER_RESHARD_FUNC(SToPReshardFunctionCrossMesh);
REGISTER_RESHARD_FUNC(RToSReshardFunction);
REGISTER_RESHARD_FUNC(RToSReshardFunctionCrossMesh);
REGISTER_RESHARD_FUNC(RToPReshardFunction);
REGISTER_RESHARD_FUNC(RToPReshardFunctionCrossMesh);
REGISTER_RESHARD_FUNC(PToRReshardFunction);
REGISTER_RESHARD_FUNC(PToRReshardFunctionCrossMesh);
REGISTER_RESHARD_FUNC(PToSReshardFunction);
REGISTER_RESHARD_FUNC(PToSReshardFunctionCrossMesh);
REGISTER_RESHARD_FUNC(SToSReshardFunction);
REGISTER_RESHARD_FUNC(SToSReshardFunctionCrossMesh);
REGISTER_RESHARD_FUNC(SameStatusReshardFunction);
REGISTER_RESHARD_FUNC(SameNdMeshReshardFunction);
REGISTER_RESHARD_FUNC(CrossNdMeshReshardFunction);

}  // namespace distributed
}  // namespace phi
