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

#include "paddle/phi/kernels/cudnn_lstm_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CudnnLSTMKernel(
    const Context& ctx,
    const DenseTensor& x,
    const DenseTensor& init_h,
    const DenseTensor& init_c,
    const paddle::optional<DenseTensor>& w,
    const paddle::optional<std::vector<const DenseTensor*>>& weight_list,
    const paddle::optional<DenseTensor>& sequence_length,
    float dropout_prob,
    bool is_bidirec,
    int hidden_size,
    int num_layers,
    bool is_test,
    int seed,
    DenseTensor* out,
    DenseTensor* last_h,
    DenseTensor* last_c,
    DenseTensor* reserve,
    DenseTensor* state_out) {
  PADDLE_THROW(common::errors::Unimplemented(
      "CPU is not support for cudnn_lstm now. Will be add in the future"));
}

}  // namespace phi

PD_REGISTER_KERNEL(cudnn_lstm, CPU, ALL_LAYOUT, phi::CudnnLSTMKernel, float) {}
