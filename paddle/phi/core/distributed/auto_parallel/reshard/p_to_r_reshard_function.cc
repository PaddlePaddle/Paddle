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

#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_r_reshard_function.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/kernels/all_reduce_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {
namespace distributed {

bool PToRReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  RESHARD_SHORTCUT_IF_FALSE(in.dist_attr().is_partial());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in.dist_attr().process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  return true;
}

void PToRReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call PToRReshardFunction Eval";
  const auto& in_dist_attr = in.dist_attr();
  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& in_process_ids = in_process_mesh.process_ids();
  const auto& in_partial_status = in_dist_attr.partial_status();
  auto in_reduce_type = in_partial_status.at(0);
  bool reduce_mean = false;
  auto dtype = in.dtype();

  if (in_reduce_type == ReduceType::kRedAvg) {
    in_reduce_type = ReduceType::kRedSum;
    reduce_mean = true;
  }
  int64_t reduce_type = static_cast<int64_t>(in_reduce_type);
  VLOG(3) << "Transfer from partial to replicated status with reduce type "
          << reduce_type;

  RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                            AllReduce,
                            dtype,
                            in_process_ids,
                            in.value(),
                            reduce_type,
                            GetMutableTensor(out));

  if (reduce_mean) {
    VLOG(3) << "Do reduce mean after all reduce sum";
    DenseTensor tensor_of_num_process;
    IntArray shape({1});
    RESHARD_FUNCTOR(dev_ctx,
                    Full,
                    in.dtype(),
                    shape,
                    static_cast<int64_t>(in_process_ids.size()),
                    &tensor_of_num_process);
    RESHARD_FUNCTOR(dev_ctx,
                    Divide,
                    dtype,
                    out->value(),
                    tensor_of_num_process,
                    GetMutableTensor(out));
  }

  SetDistProps(out, in.dims(), out_dist_attr);
}

REGISTER_RESHARD_FUNC(PToRReshardFunction);

}  // namespace distributed
}  // namespace phi
