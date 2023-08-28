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

#include "paddle/phi/core/distributed/auto_parallel/s_to_r_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_all_gather_functor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_concat_functor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_split_functor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"

namespace phi {
namespace distributed {

bool SToRReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  bool flag = true;
  const auto& in_dist_attr = in.dist_attr();

  const auto& in_dims_mapping = in_dist_attr.dims_mapping();
  const auto& out_dims_mapping = out_dist_attr.dims_mapping();

  flag &= IsDimsMappingShard(in_dims_mapping);
  flag &= IsDimsMappingReplicated(out_dims_mapping);

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  flag &= (in_process_mesh.ndim() == 1);
  flag &= (out_process_mesh.ndim() == 1);
  flag &= (in_process_mesh == out_process_mesh);

  // Ensure the tensor is balanced split, or we need send/recv rather than
  // all_gather
  std::map<int64_t, int64_t> split_axis_to_mesh_axis =
      GetSplitAxisWithDimsMapping(in_dims_mapping);
  int64_t split_axis = split_axis_to_mesh_axis.begin()->first;
  int64_t num_of_process = in_process_mesh.size();
  flag &=
      (in.local_dims()[split_axis] * num_of_process == in.dims()[split_axis]);

  return flag;
}

void SToRReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  const DenseTensor& in_physical_tensor_cur_rank = in.value();
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_dims_mapping = in_dist_attr.dims_mapping();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();

  // Since the precondition ensure the out_process_ids is equal to the
  // in_process_ids, so the participate process ids mush equal to either
  // in_process_ids or out_process_ids.
  DenseTensor out_all_gather = ReshardAllGatherFunctor(
      dev_ctx, in_physical_tensor_cur_rank, in_process_ids);

  std::map<int64_t, int64_t> split_axis_to_mesh_axis =
      GetSplitAxisWithDimsMapping(in_dims_mapping);
  int64_t split_axis = split_axis_to_mesh_axis.begin()->first;

  if (split_axis == 0) {
    // If the input dist tensor is shard(0), the subsequent split
    // and concat is unnecessary.
    set_dist_props(out, out_all_gather, out_all_gather.dims(), out_dist_attr);
  } else {
    // Since the result of all_gather always concat the tensor on axis 0,
    // first we need to split the result on axis 0,
    // then we need to concat the split result on input split axis.
    int64_t default_split_axis = 0;
    int64_t num_of_process = in_process_ids.size();

    IntArray sections(std::vector<int64_t>(
        num_of_process,
        in_physical_tensor_cur_rank.dims()[default_split_axis]));
    std::vector<DenseTensor> split_out_vec = ReshardSplitFunctor(
        *dev_ctx, out_all_gather, sections, default_split_axis);

    // Concat the result after split on correct axis.
    std::vector<const DenseTensor*> concat_input_vec;
    for (const auto& tensor : split_out_vec) {
      concat_input_vec.emplace_back(&tensor);
    }
    DenseTensor concat_out_tensor =
        ReshardConcatFunctor(*dev_ctx, concat_input_vec, split_axis);

    set_dist_props(
        out, concat_out_tensor, concat_out_tensor.dims(), out_dist_attr);
  }
}

REGISTER_RESHARD_FUNC(SToRReshardFunction);

}  // namespace distributed
}  // namespace phi
