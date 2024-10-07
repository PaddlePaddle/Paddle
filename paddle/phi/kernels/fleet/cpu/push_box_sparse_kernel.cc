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
#include <vector>
#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#endif
#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#endif
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
template <typename T, typename Context>
void PushBoxSparseKernel(const Context &dev_ctx,
                         const std::vector<const DenseTensor *> &ids,
                         const std::vector<const DenseTensor *> &out_grad,
                         bool is_sparse UNUSED,
                         bool is_distributed UNUSED,
                         int size,
                         std::vector<DenseTensor *> out_grad_out) {
  auto inputs = ids;
  auto d_output = out_grad_out;
  const auto slot_size = inputs.size();
  std::vector<const uint64_t *> all_keys(slot_size);
  std::vector<const float *> all_grad_values(slot_size);
  std::vector<int64_t> slot_lengths(slot_size);
  int batch_size = -1;
  for (size_t i = 0; i < slot_size; i++) {
    const auto *slot = inputs[i];
    const uint64_t *single_slot_keys =
        reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
    all_keys[i] = single_slot_keys;
    slot_lengths[i] = slot->numel();
    int cur_batch_size =
        slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
    if (batch_size == -1) {
      batch_size = cur_batch_size;
    } else {
      PADDLE_ENFORCE_EQ(batch_size,
                        cur_batch_size,
                        common::errors::PreconditionNotMet(
                            "The batch size of all input slots should be same, "
                            "please check"));
    }
    const float *grad_value = d_output[i]->data<float>();
    all_grad_values[i] = grad_value;
  }
#ifdef PADDLE_WITH_BOX_PS
  auto hidden_size = size;
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->PushSparseGrad(dev_ctx.GetPlace(),
                          all_keys,
                          all_grad_values,
                          slot_lengths,
                          hidden_size,
                          0,
                          batch_size);
#endif
#ifdef PADDLE_WITH_HETERPS
  auto hidden_size = size;
  auto gpu_ps_ptr = paddle::framework::PSGPUWrapper::GetInstance();
  gpu_ps_ptr->PushSparseGrad(dev_ctx.GetPlace(),
                             0,
                             all_keys,
                             all_grad_values,
                             slot_lengths,
                             hidden_size,
                             batch_size);
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(
    push_box_sparse, CPU, ALL_LAYOUT, phi::PushBoxSparseKernel, float) {}
