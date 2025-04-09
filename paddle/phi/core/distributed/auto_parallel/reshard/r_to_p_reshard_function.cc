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

#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_p_reshard_function.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi::distributed {

bool RToPReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_partial());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void RToPReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  int64_t local_rank = GetCurRankCoordInMesh(out_process_mesh)[0];
  const auto& in_reduce_type = out_dist_attr.partial_status().at(0);

  if (local_rank != 0) {
    if (in_reduce_type == ReduceType::kRedAvg) {
      // assign the input value to output
      RESHARD_FUNCTOR_WITHOUT_DTYPE(
          dev_ctx, Assign, in.value(), GetMutableTensor(out));
    } else {
      // reset the physical tensor to zero
      IntArray shape(in.local_dims().Get(), in.local_dims().size());
      RESHARD_FUNCTOR(
          dev_ctx, Full, in.dtype(), shape, 0, GetMutableTensor(out));
    }
  } else {
    // assign the input value to output
    RESHARD_FUNCTOR_WITHOUT_DTYPE(
        dev_ctx, Assign, in.value(), GetMutableTensor(out));
  }
  SetDistProps(out, in.dims(), out_dist_attr);
}

bool RToPReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_partial());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.shape() ==
                            out_process_mesh.shape());
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void RToPReshardFunctionCrossMesh::Eval(phi::DeviceContext* dev_ctx,
                                        const DistTensor& in,
                                        const TensorDistAttr& out_dist_attr,
                                        DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  DistTensor tmp_result;

  SameStatusReshardFunction same_status_func;
  TensorDistAttr tmp_dist_attr = in.dist_attr();
  tmp_dist_attr.set_process_mesh(out_process_mesh);
  same_status_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);

  int64_t cur_global_rank = GetCurGlobalRank();
  if (out_process_mesh.contains(cur_global_rank)) {
    RToPReshardFunction r_to_p_func;
    PADDLE_ENFORCE(
        r_to_p_func.IsSuitable(tmp_result, out_dist_attr),
        common::errors::InvalidArgument(
            "Invoke the r to p reshard function is not valid from %s to %s.",
            tmp_result.dist_attr(),
            out_dist_attr));
    r_to_p_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
  } else {
    SetDistProps(out, in.dims(), out_dist_attr);
    SetValue(out, tmp_result.value());
  }
}

}  // namespace phi::distributed
