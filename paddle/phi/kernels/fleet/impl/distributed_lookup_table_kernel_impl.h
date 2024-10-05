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
void DistributedLookupTableKernel(const Context &dev_ctx,
                                  const std::vector<const DenseTensor *> &ids,
                                  const DenseTensor &w,
                                  int table_id,
                                  bool is_distributed,
                                  const std::string &lookup_table_version,
                                  int64_t padding_idx,
                                  int dtype,
                                  bool is_test,
                                  std::vector<DenseTensor *> outputs) {
  int64_t emb_dim = 0;
  emb_dim = w.dims()[1];

  auto inputs = ids;
  auto fleet = paddle::distributed::FleetWrapper::GetInstance();

  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
    fleet->PullSparseToTensorSync(static_cast<uint64_t>(table_id),
                                  emb_dim,
                                  static_cast<uint64_t>(padding_idx),
                                  dev_ctx.GetPlace(),
                                  !is_test,
                                  &inputs,
                                  &outputs);
  } else {
    auto inputs_variable = ids;
    auto outputs_variable = outputs;

    auto cpu_place = phi::CPUPlace();

    std::vector<const phi::DenseTensor *> tmp_input_vec;
    auto input_var_size = inputs_variable.size();
    std::vector<phi::DenseTensor *> tmp_output_vec;
    auto output_var_size = outputs_variable.size();

    std::vector<std::shared_ptr<phi::DenseTensor>> tmp_tensors;

    // create temp input
    for (size_t idx = 0; idx < input_var_size; ++idx) {
      tmp_tensors.emplace_back(std::make_shared<phi::DenseTensor>());
      auto *p = tmp_tensors.back().get();
      phi::Copy(dev_ctx, *inputs_variable[idx], cpu_place, false, p);
      tmp_input_vec.push_back(p);
    }

    // create temp output
    for (size_t idx = 0; idx < output_var_size; ++idx) {
      tmp_tensors.emplace_back(std::make_shared<phi::DenseTensor>());
      auto *p = tmp_tensors.back().get();
      p->Resize(outputs[idx]->dims());
      tmp_output_vec.push_back(p);
    }

    // use fleet->PullSparse
    fleet->PullSparseToTensorSync(static_cast<uint64_t>(table_id),
                                  emb_dim,
                                  static_cast<uint64_t>(padding_idx),
                                  cpu_place,
                                  !is_test,
                                  &tmp_input_vec,
                                  &tmp_output_vec);

    // cp temp to origin
    for (size_t idx = 0; idx < output_var_size; ++idx) {
      dev_ctx.template Alloc<T>(outputs_variable[idx]);
      phi::Copy(dev_ctx,
                *tmp_output_vec[idx],
                dev_ctx.GetPlace(),
                false,
                outputs_variable[idx]);
    }
  }

  auto id_vars = ids;
  auto out_vars = outputs;

  if (lookup_table_version == "lookup_table_v2") {
    for (size_t i = 0; i < id_vars.size(); ++i) {
      auto *id_tensor = id_vars[i];
      auto *out_tensor = out_vars[i];

      auto id_dims = common::vectorize<int64_t>(id_tensor->dims());
      out_tensor->Resize(common::make_ddim({static_cast<int64_t>(id_dims[0]),
                                            static_cast<int64_t>(id_dims[1]),
                                            static_cast<int64_t>(emb_dim)}));
    }
  }
}
}  // namespace phi
