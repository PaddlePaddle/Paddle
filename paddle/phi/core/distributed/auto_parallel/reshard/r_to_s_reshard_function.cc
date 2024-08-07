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

#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_s_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi::distributed {

bool RToSReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void RToSReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& out_dims_mapping = out_dist_attr.dims_mapping();
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  const DenseTensor& in_physical_tensor_cur_rank = in.value();

  DenseTensor out_physical_tensor_cur_rank;

  std::map<int, int64_t> split_axis_to_mesh_axis =
      GetSplitAxisWithDimsMapping(out_dims_mapping);
  std::vector<int64_t> coord_in_mesh = GetCurRankCoordInMesh(out_process_mesh);

  int split_axis = split_axis_to_mesh_axis.begin()->first;
  int64_t mesh_axis = split_axis_to_mesh_axis.begin()->second;

  int64_t num_of_process = out_process_mesh.shape()[mesh_axis];
  VLOG(3) << "RToSReshard: Tensor will be split on axis " << split_axis
          << ". Split will use axis " << mesh_axis << " of process_mesh."
          << " There will have " << num_of_process
          << " process participate in.";

  std::vector<int64_t> split_num_vec =
      BalancedSplit(in.dims()[split_axis], num_of_process);
  IntArray sections(split_num_vec);

  std::vector<DenseTensor> split_out_vec;
  auto dtype = in_physical_tensor_cur_rank.dtype();
  RESHARD_FUNCTOR(dev_ctx,
                  Split,
                  dtype,
                  in_physical_tensor_cur_rank,
                  sections,
                  split_axis,
                  &split_out_vec);

  VLOG(3) << "The current process will remain the idx "
          << coord_in_mesh[mesh_axis] << " piece of tensor";

  SetValue(out, split_out_vec[coord_in_mesh[mesh_axis]]);
  SetDistProps(out, in.dims(), out_dist_attr);
}

bool RToSReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.shape() ==
                            out_process_mesh.shape());
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void RToSReshardFunctionCrossMesh::Eval(phi::DeviceContext* dev_ctx,
                                        const DistTensor& in,
                                        const TensorDistAttr& out_dist_attr,
                                        DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& in_dist_attr = in.dist_attr();

  DistTensor tmp_result;
  TensorDistAttr in_dist_attr_shard = in_dist_attr;
  in_dist_attr_shard.set_dims_mapping(out_dist_attr.dims_mapping());

  int64_t cur_global_rank = GetCurGlobalRank();
  if (in_dist_attr.process_mesh().contains(cur_global_rank)) {
    RToSReshardFunction r_to_s_func;
    PADDLE_ENFORCE(
        r_to_s_func.IsSuitable(in, in_dist_attr_shard),
        common::errors::InvalidArgument(
            "Invoke the r to s reshard function is not valid from %s to %s.",
            in_dist_attr,
            in_dist_attr_shard));
    r_to_s_func.Eval(dev_ctx, in, in_dist_attr_shard, &tmp_result);
  } else {
    SetDistProps(&tmp_result, in.dims(), in_dist_attr_shard);
    SetValue(&tmp_result, in.value());
  }
  SameStatusReshardFunction same_status_func;
  PADDLE_ENFORCE(
      same_status_func.IsSuitable(tmp_result, out_dist_attr),
      common::errors::InvalidArgument("Invoke the same status reshard function "
                                      "is not valid from %s to %s.",
                                      tmp_result.dist_attr(),
                                      out_dist_attr));
  same_status_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
}

}  // namespace phi::distributed
