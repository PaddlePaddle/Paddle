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
void ShuffleChannelOpCUDAKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                int group,
                                DenseTensor* out) {
  auto input_dims = x.dims();
  auto num = input_dims[0];
  auto channel = input_dims[1];
  auto height = input_dims[2];
  auto weight = input_dims[3];

  auto feature_map_size = channel * height * weight;
  auto sp_sz = height * weight;
  int group_row = group;
  int group_column = channel / group_row;
  // count is the product of NCHW same as numel()
  int count = num * group_column * group_row * sp_sz;

  int blocks = NumBlocks(out->numel());
  int threads = kNumCUDAThreads;

  const T* input_data = x.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);

  ShuffleChannel<T><<<blocks, threads, 0, dev_ctx.stream()>>>(count,
                                                              feature_map_size,
                                                              output_data,
                                                              input_data,
                                                              group_row,
                                                              group_column,
                                                              sp_sz);
}
}  // namespace phi

PD_REGISTER_KERNEL(shuffle_channel,
                   GPU,
                   ALL_LAYOUT,
                   phi::ShuffleChannelOpCUDAKernel,
                   float,
                   double) {}
