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

#include "paddle/phi/core/distributed/auto_parallel/reshard/x_to_r_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi {
namespace distributed {

bool XToRReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);

  return true;
}

void XToRReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call XToRReshardFunction Eval";
  const auto& in_dist_attr = in.dist_attr();

  DistTensor tmp_result;
  TensorDistAttr tmp_dist_attr = in_dist_attr;
  if (in_dist_attr.is_shard()) {
    // if 'x' is 's', invoke s2r
    SToRReshardFunction s_to_r_func;
    PADDLE_ENFORCE(
        s_to_r_func.IsSuitable(in, tmp_dist_attr),
        phi::errors::InvalidArgument(
            "Invoke the s to r reshard function is not valid from %s to %s.",
            in_dist_attr,
            tmp_dist_attr));
    s_to_r_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);
  } else if (in_dist_attr.is_partial()) {
    // if 'x' is 'p', invoke p2r
    PToRReshardFunction p_to_r_func;
    PADDLE_ENFORCE(
        p_to_r_func.IsSuitable(in, tmp_dist_attr),
        phi::errors::InvalidArgument(
            "Invoke the p to r reshard function is not valid from %s to %s.",
            in_dist_attr,
            tmp_dist_attr));
    p_to_r_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);
  } else {
    // if 'x' is 'r', do nothing
  }
  // send to out mesh
  SameStatusReshardFunction same_status_func;
  same_status_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
}

}  // namespace distributed
}  // namespace phi
