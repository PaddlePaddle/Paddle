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

#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/reduce_scatter_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace distributed {

bool PToSReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_partial());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void PToSReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call PToSReshardFunction Eval";
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();
  auto dtype = in.dtype();
  const auto& logical_ddim = in.dims();

  int out_split_axis =
      GetSplitAxisWithDimsMapping(out_dist_attr.dims_mapping()).begin()->first;

  DenseTensor in_reduce_scatter = in.value();
  std::vector<int> axis;
  if (out_split_axis != 0) {
    for (size_t i = 0; i < common::vectorize(logical_ddim).size(); ++i) {
      axis.emplace_back(i);
    }
    std::swap(axis[0], axis[out_split_axis]);
    VLOG(3) << "out_split_axis: " << out_split_axis;
    for (size_t i = 0; i < axis.size(); i++)
      VLOG(3) << "axis[" << i << "]: " << axis[i];
    RESHARD_FUNCTOR(
        dev_ctx, Transpose, dtype, in.value(), axis, &in_reduce_scatter);
  }

  for (size_t i = 0; i < in_process_ids.size(); i++)
    VLOG(3) << "in_process_ids[" << i << "]: " << in_process_ids[i];

  DenseTensor out_reduce_scatter;
  RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                            ReduceScatter,
                            dtype,
                            in_process_ids,
                            in_reduce_scatter,
                            static_cast<int64_t>(in_process_ids.size()),
                            &out_reduce_scatter);

  if (out_split_axis != 0) {
    RESHARD_FUNCTOR(dev_ctx,
                    Transpose,
                    dtype,
                    out_reduce_scatter,
                    axis,
                    GetMutableTensor(out));
  } else {
    SetValue(out, out_reduce_scatter);
  }

  SetDistProps(out, in.dims(), out_dist_attr);
}

bool PToSReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_partial());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_shard());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void PToSReshardFunctionCrossMesh::Eval(phi::DeviceContext* dev_ctx,
                                        const DistTensor& in,
                                        const TensorDistAttr& out_dist_attr,
                                        DistTensor* out) {
  VLOG(3) << "Call PToSReshardFunctionCrossMesh Eval";
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  DistTensor tmp_result;

  SameStatusReshardFunction same_status_func;
  TensorDistAttr tmp_dist_attr = in.dist_attr();
  tmp_dist_attr.set_process_mesh(out_process_mesh);
  same_status_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);

  int64_t cur_global_rank = phi::distributed::GetCurGlobalRank();
  if (out_process_mesh.contains(cur_global_rank)) {
    PToSReshardFunction p_to_s_func;
    PADDLE_ENFORCE(
        p_to_s_func.IsSuitable(tmp_result, out_dist_attr),
        phi::errors::InvalidArgument(
            "Invoke the p to r reshard function is not valid from %s to %s.",
            tmp_result.dist_attr(),
            out_dist_attr));
    p_to_s_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
  } else {
    SetDistProps(out, in.dims(), out_dist_attr);
    SetValue(out, tmp_result.value());
  }
  // p_to_r_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
}

}  // namespace distributed
}  // namespace phi
