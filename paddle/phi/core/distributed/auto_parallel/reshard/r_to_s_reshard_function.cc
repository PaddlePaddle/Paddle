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
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

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

namespace {
std::map<int, int64_t> GetSplitAxisWithDimsMapping(
    const std::vector<std::vector<int64_t>>& dims_mapping) {
  std::map<int, int64_t> split_axis_to_mesh_axis;
  for (size_t i = 0; i < dims_mapping.size(); ++i) {
    if (dims_mapping[i].size() > 0) {
      split_axis_to_mesh_axis.emplace(i, dims_mapping[i][0]);
    }
  }
  return split_axis_to_mesh_axis;
}
}  // namespace

void RToSReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& out_dims_mapping = out_dist_attr.multi_dims_mapping();
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  const DenseTensor& in_physical_tensor_cur_rank = in.value();

  std::map<int, int64_t> split_axis_to_mesh_axis =
      GetSplitAxisWithDimsMapping(out_dims_mapping);
  std::vector<int64_t> coord_in_mesh = GetCurRankCoordInMesh(out_process_mesh);

  int split_axis = split_axis_to_mesh_axis.begin()->first;
  int64_t mesh_axis = split_axis_to_mesh_axis.begin()->second;

  VLOG(3) << "split axis is " << split_axis << ", mesh axis is " << mesh_axis;
  VLOG(3) << "shape size is " << out_process_mesh.shape().size();

  int64_t num_of_process = out_process_mesh.shape()[mesh_axis];
  if (num_of_process == 1) {
    SetValue(out, in.value());
    SetDistProps(out, in.dims(), out_dist_attr);
    return;
  }
  VLOG(3) << "RToSReshard: Tensor will be split on axis " << split_axis
          << ". Slice will use axis " << mesh_axis << " of process_mesh."
          << " There will have " << num_of_process
          << " process participate in.";

  int64_t num_group = out_dist_attr.get_split_factor(mesh_axis);
  VLOG(3) << "num group = " << num_group;

  std::vector<int64_t> split_num_vec =
      BalancedSplit(in.value().dims()[split_axis], num_of_process * num_group);

  auto dtype = in_physical_tensor_cur_rank.dtype();

  int64_t slice_stride = num_of_process;
  std::vector<DenseTensor> dense_out_vec(num_group);
  for (int64_t i = 0; i < num_group; i++) {
    int64_t start =
        split_num_vec[0] * (coord_in_mesh[mesh_axis] + i * slice_stride);
    int64_t end = std::min(start + split_num_vec[0],
                           in_physical_tensor_cur_rank.dims()[split_axis]);
    VLOG(4) << "start is " << start << ", end is " << end;

    PADDLE_ENFORCE_LE(start,
                      end,
                      common::errors::InvalidArgument(
                          "Slice Args 'start' should be less or qual to 'end', "
                          "but got 'start' is %d, 'end' is %d.",
                          start,
                          end));
    RESHARD_FUNCTOR(dev_ctx,
                    Slice,
                    dtype,
                    in_physical_tensor_cur_rank,
                    {split_axis},
                    {start},
                    {end},
                    &(dense_out_vec[i]));
  }

  if (num_group == 1) {
    SetValue(out, dense_out_vec[0]);
  } else {
    std::vector<const DenseTensor*> d_tensor_ptr_vec;
    for (auto& d_tensor : dense_out_vec) {
      d_tensor_ptr_vec.push_back(&d_tensor);
    }

    DenseTensor dense_out;
    RESHARD_FUNCTOR(dev_ctx,
                    Concat,
                    dtype,
                    d_tensor_ptr_vec,
                    Scalar(split_axis),
                    &dense_out);
    SetValue(out, dense_out);
  }
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
