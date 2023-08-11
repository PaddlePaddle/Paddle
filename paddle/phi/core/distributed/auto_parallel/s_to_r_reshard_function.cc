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
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_all_gather_functor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"

namespace phi {
namespace distributed {

bool SToRReshardFunction::IsSuitable(
    const DistTensor& in,
    const std::shared_ptr<TensorDistAttr>& out_dist_attr) {
  bool flag = true;
  const auto& in_dist_attr = in.dist_attr();

  const auto& in_dims_mapping = in_dist_attr->dims_mapping();
  const auto& out_dims_mapping = out_dist_attr->dims_mapping();

  flag &= IsDimsMappingShard(in_dims_mapping);
  flag &= IsDimsMappingReplicated(out_dims_mapping);

  const auto& in_process_mesh = in_dist_attr->process_mesh();
  const auto& out_process_mesh = out_dist_attr->process_mesh();

  flag &= (in_process_mesh.ndim() == 1);
  flag &= (out_process_mesh.ndim() == 1);
  flag &= (in_process_mesh == out_process_mesh);

  return flag;
}

std::shared_ptr<DistTensor> SToRReshardFunction::Eval(
    DeviceContext* dev_ctx,
    const DistTensor& in,
    const std::shared_ptr<TensorDistAttr>& out_dist_attr) {
  // TODO(liyurui): Only support transfer shard(0) to replicate for now.
  // Concat is needed when transfer shard(x) to replicate, will be supported
  // later.
  const DenseTensor& in_physical_tensor_cur_rank = in.value();
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_process_mesh = in_dist_attr->process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();

  // Since the precondition ensure the out_process_ids is equal to the
  // in_process_ids, so the participate process ids mush equal to either
  // in_process_ids or out_process_ids.
  DenseTensor out_all_gather = ReshardAllGatherFunctor(
      dev_ctx, in_physical_tensor_cur_rank, in_process_ids);

  return std::make_shared<DistTensor>(
      std::make_shared<DenseTensor>(out_all_gather), out_dist_attr);
}

}  // namespace distributed
}  // namespace phi
