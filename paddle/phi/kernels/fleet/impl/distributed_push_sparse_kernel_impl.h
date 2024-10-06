// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void DistributedPushSparseKernel(const Context &dev_ctx,
                                 const std::vector<const DenseTensor *> &ids,
                                 const DenseTensor &shows_in,
                                 const DenseTensor &clicks_in,
                                 int table_id,
                                 int size,
                                 bool is_distributed,
                                 const std::string &push_sparse_version,
                                 int64_t padding_idx,
                                 int dtype,
                                 bool is_test,
                                 bool use_cvm_op,
                                 const std::vector<int> &slots,
                                 std::vector<DenseTensor *> outputs) {
  auto emb_dim = size;

  auto inputs = ids;
  auto shows = &shows_in;
  auto clks = &clicks_in;

  auto fleet = paddle::distributed::FleetWrapper::GetInstance();

  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
    fleet->PushSparseFromTensorAsync(static_cast<uint64_t>(table_id),
                                     emb_dim,
                                     static_cast<uint64_t>(padding_idx),
                                     dev_ctx.GetPlace(),
                                     &inputs,
                                     slots,
                                     shows,
                                     clks,
                                     &outputs,
                                     use_cvm_op);
  } else {
    auto inputs_variable = ids;
    auto outputs_variable = outputs;

    auto cpu_place = phi::CPUPlace();

    std::vector<const phi::DenseTensor *> tmp_input_vec;
    auto input_var_size = inputs_variable.size();
    std::vector<phi::DenseTensor *> tmp_output_vec;
    auto output_var_size = outputs_variable.size();

    // create temp input
    for (size_t idx = 0; idx < input_var_size; ++idx) {
      phi::DenseTensor tmp_input_tensor;
      phi::Copy(
          dev_ctx, *inputs_variable[idx], cpu_place, false, &tmp_input_tensor);
      tmp_input_vec.push_back(&tmp_input_tensor);
    }

    phi::DenseTensor tmp_shows_tensor;
    phi::DenseTensor tmp_clicks_tensor;
    phi::Copy(dev_ctx, *shows, cpu_place, false, &tmp_shows_tensor);
    phi::Copy(dev_ctx, *clks, cpu_place, false, &tmp_clicks_tensor);

    // create temp output
    for (size_t idx = 0; idx < output_var_size; ++idx) {
      phi::DenseTensor tmp_output_tensor;
      tmp_output_tensor.Resize(outputs[idx]->dims());
      tmp_output_vec.push_back(&tmp_output_tensor);
    }

    // use fleet->PullSparse
    fleet->PushSparseFromTensorAsync(static_cast<uint64_t>(table_id),
                                     emb_dim,
                                     static_cast<uint64_t>(padding_idx),
                                     dev_ctx.GetPlace(),
                                     &tmp_input_vec,
                                     slots,
                                     &tmp_shows_tensor,
                                     &tmp_clicks_tensor,
                                     &tmp_output_vec);

    // cp temp to origin
    for (size_t idx = 0; idx < output_var_size; ++idx) {
      phi::Copy(dev_ctx,
                *tmp_output_vec[idx],
                dev_ctx.GetPlace(),
                false,
                outputs_variable[idx]);
    }
  }
}
}  // namespace phi
