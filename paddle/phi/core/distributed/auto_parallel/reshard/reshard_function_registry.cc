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

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/global_and_sub_mesh_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_x_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/x_to_r_reshard_function.h"

namespace phi::distributed {

ReshardFunction* ChooseProperReshardFunction(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  for (const auto& func : GetReshardFunctionList()) {
    if (func->IsSuitable(in, out_dist_attr)) {
      VLOG(4) << "Choose ReshardFunction: " << func->Name();
      return func.get();
    }
  }
  PADDLE_THROW(common::errors::Unimplemented(
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
REGISTER_RESHARD_FUNC(XToRShrinkReshardFunction);
REGISTER_RESHARD_FUNC(RToXExpandReshardFunction);
REGISTER_RESHARD_FUNC(SameStatusReshardFunction);
REGISTER_RESHARD_FUNC(SameNdMeshReshardFunction);
REGISTER_RESHARD_FUNC(CrossNdMeshReshardFunction);
REGISTER_RESHARD_FUNC(GlobalToSubMeshReshardFunction);
REGISTER_RESHARD_FUNC(SubMeshToGlobalReshardFunction);

}  // namespace phi::distributed
