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

#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/kernels/reduce_scatter_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace distributed {

bool SToPReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_shard());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_partial());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void SToPReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call SToPReshardFunction Eval";

  // step 1, create tmp dist attr and tmp dist tensor
  TensorDistAttr tmp_attr(out_dist_attr);
  DistTensor tmp_tensor;
  tmp_attr.clean_partial_status();

  // step 2, do s to r reshard on `in` to `tmp`
  SToRReshardFunction s_to_r;
  s_to_r.Eval(dev_ctx, in, tmp_attr, &tmp_tensor);

  // step 3, do r to p reshard on `tmp` to `out`
  RToPReshardFunction r_to_p;
  r_to_p.Eval(dev_ctx, tmp_tensor, out_dist_attr, out);
}

}  // namespace distributed
}  // namespace phi
