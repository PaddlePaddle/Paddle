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

#include "paddle/phi/core/distributed/auto_parallel/r_to_p_reshard_function.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {
namespace distributed {

bool RToPReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  bool flag = true;
  const auto& in_dist_attr = in.dist_attr();

  flag &= in_dist_attr.is_replicated();
  flag &= out_dist_attr.is_partial();

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  flag &= (in_process_mesh.ndim() == 1);
  flag &= (out_process_mesh.ndim() == 1);
  flag &= (in_process_mesh == out_process_mesh);

  return flag;
}

void RToPReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call RToPReshardFunction Eval";
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  int64_t local_rank = GetCurRankCoordInMesh(out_process_mesh)[0];
  IntArray shape(in.dims().Get(), in.dims().size());

  if (local_rank != 0) {
    // reset the physical tensor to zero
    RESHARD_FUNCTOR(dev_ctx, Full, in.dtype(), shape, 0, GetMutableTensor(out));
  } else {
    // assign the input value to output
    RESHARD_FUNCTOR_WITHOUT_DTYPE(
        dev_ctx, Assign, in.value(), GetMutableTensor(out));
  }
  SetDistProps(out, in.dims(), out_dist_attr);
}

REGISTER_RESHARD_FUNC(RToPReshardFunction);

}  // namespace distributed
}  // namespace phi
