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
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/kernels/all_reduce_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {
namespace distributed {

bool PToRReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  VLOG(0) << "PToRReshardFunction::IsSuitable in dist_attr"
          << in.dist_attr().to_string();
  VLOG(0) << "PToRReshardFunction::IsSuitable out dist_attr"
          << out_dist_attr.to_string();
  VLOG(0) << "in is partial " << in.dist_attr().is_partial();
  VLOG(0) << "out is replicated " << out_dist_attr.is_replicated();
  RESHARD_SHORTCUT_IF_FALSE(in.dist_attr().is_partial());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in.dist_attr().process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  VLOG(0) << "in mesh " << in_process_mesh.to_string();
  VLOG(0) << "out mesh " << out_process_mesh.to_string();
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh == out_process_mesh);

  VLOG(0) << "is suitable return true";
  return true;
}

void PToRReshardFunction::Eval(DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call " << Name();
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
  int reduce_type = static_cast<int>(in_reduce_type);
  VLOG(3) << "Transfer from partial to replicated status with reduce type "
          << reduce_type;

  if (dev_ctx) {
    RESHARD_FUNCTOR_WITH_COMM(dev_ctx,
                              AllReduce,
                              dtype,
                              in_process_ids,
                              in.value(),
                              reduce_type,
                              GetMutableTensor(out));
  } else {
    reshard_func_descs_.emplace_back(
        std::make_unique<AllReduceOpDesc>(dtype, in_process_ids, reduce_type));
  }

  if (reduce_mean) {
    VLOG(3) << "Do reduce mean after all reduce sum";
    DenseTensor tensor_of_num_process;
    IntArray shape({1});
    if (dev_ctx) {
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
    } else {
      // reshard_func_descs_.emplace_back(std::make_unique<FullOpDesc>(dtype,
      // shape, static_cast<int64_t>(in_process_ids.size())));
      // reshard_func_descs_.emplace_back(std::make_unique<DivideOpDesc>(dtype));
    }
  }

  SetDistProps(out, in.dims(), out_dist_attr);
}

bool PToRReshardFunctionCrossMesh::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const auto& in_dist_attr = in.dist_attr();

  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_partial());
  RESHARD_SHORTCUT_IF_FALSE(out_dist_attr.is_replicated());

  const auto& in_process_mesh = in_dist_attr.process_mesh();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  VLOG(0) << "cross_mesh in mesh " << in_process_mesh.to_string();
  VLOG(0) << "cross_mesh out mesh " << out_process_mesh.to_string();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(out_process_mesh.ndim() == 1);
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.shape() ==
                            out_process_mesh.shape());
  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh != out_process_mesh);

  return true;
}

void PToRReshardFunctionCrossMesh::Eval(phi::DeviceContext* dev_ctx,
                                        const DistTensor& in,
                                        const TensorDistAttr& out_dist_attr,
                                        DistTensor* out) {
  VLOG(3) << "Call " << Name();
  const auto& out_process_mesh = out_dist_attr.process_mesh();

  DistTensor tmp_result;

  SameStatusReshardFunction same_status_func;
  TensorDistAttr tmp_dist_attr = in.dist_attr();
  tmp_dist_attr.set_process_mesh(out_process_mesh);
  same_status_func.Eval(dev_ctx, in, tmp_dist_attr, &tmp_result);

  VLOG(0) << "find same_status_func func_size " << reshard_func_descs_.size();

  int64_t cur_global_rank = GetCurGlobalRank();
  if (out_process_mesh.contains(cur_global_rank)) {
    VLOG(0) << "out_process_mesh contains " << cur_global_rank;
    PToRReshardFunction p_to_r_func;
    PADDLE_ENFORCE(
        p_to_r_func.IsSuitable(tmp_result, out_dist_attr),
        phi::errors::InvalidArgument(
            "Invoke the p to r reshard function is not valid from %s to %s.",
            tmp_result.dist_attr(),
            out_dist_attr));
    p_to_r_func.Eval(dev_ctx, tmp_result, out_dist_attr, out);
  } else {
    VLOG(0) << "out_process_mesh not contains " << cur_global_rank;
    SetDistProps(out, in.dims(), out_dist_attr);
    SetValue(out, tmp_result.value());
  }
}

}  // namespace distributed
}  // namespace phi
