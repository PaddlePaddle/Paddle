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

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/shuffle_channel.h"

namespace phi {

template <typename T, typename Context>
void ShuffleChannelGradOpCUDAKernel(const Context& dev_ctx,
                                    const DenseTensor& out_grad,
                                    int group,
                                    DenseTensor* x_grad) {
  const auto& input_dims = x_grad->dims();
  auto num = input_dims[0];
  auto channel = input_dims[1];
  auto height = input_dims[2];
  auto weight = input_dims[3];
  auto feature_map_size = channel * height * weight;
  auto sp_sz = height * weight;

  int group_row = group;
  int group_column = channel / group_row;

  T* input_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* output_grad_data = out_grad.data<T>();

  int blocks = NumBlocks(out_grad.numel());
  int threads = kNumCUDAThreads;
  int count = num * group_column * group_row * sp_sz;

  ShuffleChannel<T><<<blocks, threads, 0, dev_ctx.stream()>>>(count,
                                                              feature_map_size,
                                                              input_grad_data,
                                                              output_grad_data,
                                                              group_row,
                                                              group_column,
                                                              sp_sz);
}
}  // namespace phi

PD_REGISTER_KERNEL(shuffle_channel_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ShuffleChannelGradOpCUDAKernel,
                   float,
                   double) {}
