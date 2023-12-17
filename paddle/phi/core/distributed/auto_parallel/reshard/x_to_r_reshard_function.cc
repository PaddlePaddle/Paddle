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
  const auto& in_mesh = in_dist_attr.process_mesh();

  DistTensor tmp_result;
  TensorDistAttr tmp_dist_attr = in_dist_attr;
  if (in_dist_attr.is_shard()) {
    // if 'x' is 's', invoke in-mesh s2r
    // if (in_mesh.contains(cur_global_rank)) {
    SToRReshardFunction s_to_r_func;
    s_to_r_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);
    // } else {
    //   SetDistProps(&tmp_result, in.dims(), tmp_dist_attr);
    //   SetValue(&tmp_result, in.value());
    // }
  } else if (in_dist_attr.is_partial()) {
    // if 'x' is 'p', invoke p2r
    // if (in_mesh.contains(cur_global_rank)) {
    PToRReshardFunction p_to_r_func;
    p_to_r_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);
    // } else {
    //   SetDistProps(&tmp_result, in.dims(), tmp_dist_attr);
    //   SetValue(&tmp_result, in.value());
    // }
  } else {
    // if 'x' is 'r', just copy
    tmp_result = in;
  }

  // send to out mesh
  // set tmp_result mesh
  TensorDistAttr tmp_dist_attr_send = tmp_result.dist_attr();
  const auto tmp_mesh_send =
      ProcessMesh({1}, {in_mesh.process_ids()[0]}, {in_mesh.dim_names()[0]});
  tmp_dist_attr_send.set_process_mesh(tmp_mesh_send);
  SetDistProps(&tmp_result, tmp_dist_attr_send);

  // if out_mesh is same as in_mesh[0], do not need to do p2p comm
  if (out_dist_attr.process_mesh() != tmp_mesh_send) {
    SameStatusReshardFunction same_status_func;
    same_status_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
  } else {
    SetDistProps(out, tmp_result.dims(), out_dist_attr);
    SetValue(out, tmp_result.value());
  }
}

}  // namespace distributed
}  // namespace phi
