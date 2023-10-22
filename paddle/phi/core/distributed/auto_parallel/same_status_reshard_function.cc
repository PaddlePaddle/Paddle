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

#include "paddle/phi/core/distributed/auto_parallel/same_status_reshard_function.h"

#include <algorithm>

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/p_recv_kernel.h"
#include "paddle/phi/kernels/p_send_kernel.h"

namespace phi {
namespace distributed {

namespace {

std::vector<int64_t> GetUnionProcessIds(std::vector<int64_t> in_process_ids,
                                        std::vector<int64_t> out_process_ids) {
  std::vector<int64_t> result;
  std::sort(in_process_ids.begin(), in_process_ids.end());
  std::sort(out_process_ids.begin(), out_process_ids.end());
  std::set_union(in_process_ids.begin(),
                 in_process_ids.end(),
                 out_process_ids.begin(),
                 out_process_ids.end(),
                 std::back_inserter(result));
  return result;
}

}  // namespace

bool SameStatusReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  bool flag = true;
  const auto& in_dist_attr = in.dist_attr();

  flag &= (in_dist_attr.dims_mapping() == out_dist_attr.dims_mapping());
  flag &= (in_dist_attr.partial_dims() == out_dist_attr.partial_dims());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  flag &= (in_process_mesh != out_process_mesh);
  flag &= (in_process_mesh.shape() == out_process_mesh.shape());

  return flag;
}

void SameStatusReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                     const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr,
                                     DistTensor* out) {
  VLOG(3) << "Call SameStatusReshardFunction Eval";
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();
  const auto& out_process_mesh = out_dist_attr.process_mesh();
  const auto& out_process_ids = out_process_mesh.process_ids();
  auto all_process_ids = GetUnionProcessIds(in_process_ids, out_process_ids);
  auto dtype = in.dtype();
  // TODO(liyurui): Use dynamic shape will lead to poor performance, but we
  // don't have any other good idea now. For the following reasons:
  // 1. We can not ensure the meta being right deduce by the infermeta.
  // 2. The meta of some kernels can't decide in compile time.
  // 3. DenseTensor with empty value only need infermeta and skip the real
  // kernel execution.
  bool dynamic_shape = true;

  // TODO(GhostScreaming): After cross-mesh reshard, current device may
  // needs to execute next layer. When it construct next layer's backward
  // graph, out->place() will be called such as in SetGradOutMeta method. As
  // a result, out can't be undefined. Try to allocate a zero-memory value
  // for out. Following send/recv will cover this empty DenseTensor
  // construction.
  VLOG(3) << "Same_status_reshard_function create an empty DenseTensor for "
             "cross-mesh DistTensor.";
  *(out->unsafe_mutable_value()) =
      phi::DenseTensor(std::make_shared<phi::Allocation>(
                           nullptr, 0, phi::distributed::GetDefaultPlace()),
                       in.value().meta());

  std::vector<std::pair<int64_t, int64_t>> p2p_pair;
  for (size_t i = 0; i < out_process_ids.size(); ++i) {
    p2p_pair.emplace_back(
        std::make_pair(in_process_ids[i], out_process_ids[i]));
  }

  int64_t cur_global_rank = GetCurGlobalRank();
  for (const auto& iter : p2p_pair) {
    int64_t src = iter.first;
    int64_t dst = iter.second;
    if (src == cur_global_rank) {
      VLOG(3) << "Send from src " << src << " to dst " << dst;
      int64_t dst_local_rank = GetLocalRankInParticipate(all_process_ids, dst);
      // Sice send kernel only has input, so we don't need to infermeta
      // actually. According to this reason, just use the kernel directly.
      RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                PSendKernel,
                                dtype,
                                all_process_ids,
                                in.value(),
                                dst_local_rank,
                                dynamic_shape);
    } else if (dst == cur_global_rank) {
      VLOG(3) << "Recv from src " << src << " to dst " << dst;
      int64_t src_local_rank = GetLocalRankInParticipate(all_process_ids, src);
      RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                                PRecv,
                                dtype,
                                all_process_ids,
                                src_local_rank,
                                dynamic_shape,
                                GetMutableTensor(out));
    }
  }
  SetDistProps(out, in.dims(), out_dist_attr);
}

REGISTER_RESHARD_FUNC(SameStatusReshardFunction);

}  // namespace distributed
}  // namespace phi
