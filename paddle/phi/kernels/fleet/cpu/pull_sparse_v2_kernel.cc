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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void PullSparseV2Kernel(const Context &dev_ctx,
                        const std::vector<const DenseTensor *> &ids,
                        const std::vector<const DenseTensor *> &w,
                        int embedding_dim,
                        int table_id_in,
                        const std::string &accessor_class,
                        const std::string &ctr_label_name,
                        int padding_id_in,
                        bool scale_sparse_grad,
                        const std::vector<std::string> &input_names,
                        bool is_distributed,
                        std::vector<DenseTensor *> out) {
  auto inputs = ids;
  auto outputs = out;
  uint32_t fea_dim = static_cast<uint32_t>(embedding_dim);
  uint64_t padding_id = static_cast<uint64_t>(padding_id_in);
  auto table_id = static_cast<uint32_t>(table_id_in);
  // note: GetInstance() is not thread-safe
  // we assume FleetWrapper has been already initialized
  auto fleet_ptr = paddle::framework::FleetWrapper::GetInstance();
  fleet_ptr->PullSparseToTensorSync(
      table_id, fea_dim, padding_id, dev_ctx.GetPlace(), &inputs, &outputs);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    pull_sparse_v2, CPU, ALL_LAYOUT, phi::PullSparseV2Kernel, float) {}
