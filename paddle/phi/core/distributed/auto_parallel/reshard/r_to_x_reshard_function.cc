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

#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_x_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/add_n_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/p_recv_kernel.h"
#include "paddle/phi/kernels/p_send_kernel.h"

namespace phi {
namespace distributed {

bool RToXExpandReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.process_ids().size() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.process_ids().size() != 1);

  return true;
}

void RToXExpandReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                     const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr,
                                     DistTensor* out) {
  VLOG(3) << "Call RToXExpandReshardFunction Eval";
}

}  // namespace distributed
}  // namespace phi
