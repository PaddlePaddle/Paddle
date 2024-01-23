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

#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/all_gather_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi {
namespace distributed {

bool SToRReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_dims_mapping = in_dist_attr.dims_mapping();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_shard());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  // Ensure the tensor is balanced split, or we need send/recv rather than
  // all_gather
  int split_axis = GetSplitAxisWithDimsMapping(in_dims_mapping).begin()->first;
  int64_t num_of_process = in_process_mesh.size();
  RESHARD_SHORTCUT_IF_FALSE(in.local_dims()[static_cast<int>(split_axis)] *
                                num_of_process ==
                            in.dims()[static_cast<int>(split_axis)]);

  return true;
}

void SToRReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call SToRReshardFunction Eval";
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_dims_mapping = in_dist_attr.dims_mapping();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();
  auto dtype = in.dtype();

  // Since the precondition ensure the out_process_ids is equal to the
  // in_process_ids, so the participate process ids mush equal to either
  // in_process_ids or out_process_ids.
  RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                            AllGather,
                            dtype,
                            in_process_ids,
                            in.value(),
                            in_process_ids.size(),
                            GetMutableTensor(out));
  int split_axis = GetSplitAxisWithDimsMapping(in_dims_mapping).begin()->first;

  if (split_axis == 0) {
    // If the input dist tensor is shard(0), the subsequent split
    // and concat is unnecessary.
    SetDistProps(out, in.dims(), out_dist_attr);
  } else {
    // Since the result of all_gather always concat the tensor on axis 0,
    // first we need to split the result on axis 0,
    // then we need to concat the split result on input split axis.
    int64_t default_split_axis = 0;
    int64_t num_of_process = static_cast<int64_t>(in_process_ids.size());

    IntArray sections(std::vector<int64_t>(
        num_of_process,
        in.value().dims()[static_cast<int>(default_split_axis)]));
    std::vector<DenseTensor> split_out_vec;
    RESHARD_FUNCTOR(dev_ctx,
                    Split,
                    dtype,
                    out->value(),
                    sections,
                    default_split_axis,
                    &split_out_vec);

    // Concat the result after split on correct axis.
    std::vector<const DenseTensor*> concat_input_vec;
    for (const auto& tensor : split_out_vec) {
      concat_input_vec.emplace_back(&tensor);
    }

    RESHARD_FUNCTOR(dev_ctx,
                    Concat,
                    dtype,
                    concat_input_vec,
                    split_axis,
                    GetMutableTensor(out));

    SetDistProps(out, in.dims(), out_dist_attr);
  }
}

bool SToRReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_dims_mapping = in_dist_attr.dims_mapping();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_shard());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  int64_t cur_global_rank = GetCurGlobalRank();
  if (in_process_mesh.contains(cur_global_rank)) {
    int split_axis =
        GetSplitAxisWithDimsMapping(in_dims_mapping).begin()->first;
    int64_t num_of_process = in_process_mesh.size();
    RESHARD_SHORTCUT_IF_FALSE(in.local_dims()[static_cast<int>(split_axis)] *
                                  num_of_process ==
                              in.dims()[static_cast<int>(split_axis)]);
  }

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.shape() ==
                            out_process_mesh.shape());
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void SToRReshardFunctionCrossMesh::Eval(DeviceContext* dev_ctx,
                                        const DistTensor& in,
                                        const TensorDistAttr& out_dist_attr,
                                        DistTensor* out) {
  VLOG(3) << "Call SToRReshardFunctionCrossMesh Eval";
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  SameStatusReshardFunction same_status_func;
  DistTensor tmp_result;

  TensorDistAttr tmp_dist_attr = in.dist_attr();
  tmp_dist_attr.set_process_mesh(out_process_mesh);
  same_status_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);

  int64_t cur_global_rank = GetCurGlobalRank();
  if (out_process_mesh.contains(cur_global_rank)) {
    SToRReshardFunction s_to_r_func;
    PADDLE_ENFORCE(
        s_to_r_func.IsSuitable(tmp_result, out_dist_attr),
        phi::errors::InvalidArgument(
            "Invoke the s to r reshard function is not valid from %s to %s.",
            tmp_result.dist_attr(),
            out_dist_attr));
    s_to_r_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
  } else {
    SetDistProps(out, in.dims(), out_dist_attr);
    SetValue(out, tmp_result.value());
  }
}

}  // namespace distributed
}  // namespace phi
