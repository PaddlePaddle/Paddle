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

#include "paddle/phi/core/distributed/auto_parallel/r_to_s_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_split_functor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/core/kernel_factory.h"

namespace phi {
namespace distributed {

bool RToSReshardFunction::IsSuitable(
    const DistTensor& in,
    const std::shared_ptr<TensorDistAttr>& out_dist_attr) {
  bool flag = true;
  const auto& in_dist_attr = in.dist_attr();

  const auto& in_dims_mapping = in_dist_attr->dims_mapping();
  const auto& out_dims_mapping = out_dist_attr->dims_mapping();

  flag &= IsDimsMappingReplicated(in_dims_mapping);
  flag &= IsDimsMappingShard(out_dims_mapping);

  const auto& in_process_mesh = in_dist_attr->process_mesh();
  const auto& out_process_mesh = out_dist_attr->process_mesh();

  flag &= (in_process_mesh.ndim() == 1);
  flag &= (out_process_mesh.ndim() == 1);
  flag &= (in_process_mesh == out_process_mesh);

  return flag;
}

std::shared_ptr<DistTensor> RToSReshardFunction::Eval(
    const phi::DeviceContext& dev_ctx,
    const DistTensor& in,
    const std::shared_ptr<TensorDistAttr>& out_dist_attr) {
  const auto& out_dims_mapping = out_dist_attr->dims_mapping();
  const auto& out_process_mesh = out_dist_attr->process_mesh();
  const DenseTensor& in_physical_tensor_cur_rank = in.value();

  DenseTensor out_physical_tensor_cur_rank;

  std::map<int64_t, int64_t> split_axis_to_mesh_axis =
      GetSplitAxisWithDimsMapping(out_dims_mapping);
  std::vector<int64_t> coord_in_mesh = GetCurRankCoordInMesh(out_process_mesh);

  int64_t split_axis = split_axis_to_mesh_axis.begin()->first;
  int64_t mesh_axis = split_axis_to_mesh_axis.begin()->second;

  PADDLE_ENFORCE_LT(
      mesh_axis,
      out_process_mesh.ndim(),
      phi::errors::OutOfRange(
          "The mesh axis %lld exceed the size of process mesh %lld.",
          mesh_axis,
          out_process_mesh.ndim()));

  int64_t num_of_process = out_process_mesh.shape()[mesh_axis];
  VLOG(3) << "RToSReshard: Tensor will be split on axis " << split_axis
          << ". Split will use axis " << mesh_axis << " of process_mesh."
          << " There will have " << num_of_process
          << " process participate in.";

  // TODO(liyurui): Consider the tensor can not be balanced split,
  // for example, the shape of tensor is {6} but want to split it by 4
  // process.
  IntArray sections(std::vector<int64_t>(
      num_of_process, in.dims()[split_axis] / num_of_process));

  std::vector<DenseTensor> split_out_vec = ReshardSplitFunctor(
      dev_ctx, in_physical_tensor_cur_rank, sections, split_axis);

  VLOG(3) << "The current process will remain the idx "
          << coord_in_mesh[mesh_axis] << " piece of tensor";
  out_physical_tensor_cur_rank = split_out_vec[coord_in_mesh[mesh_axis]];

  return std::make_shared<DistTensor>(
      std::make_shared<DenseTensor>(out_physical_tensor_cur_rank),
      out_dist_attr);
}

}  // namespace distributed
}  // namespace phi
