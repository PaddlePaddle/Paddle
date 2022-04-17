// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/nll_loss_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/nll_loss.h"

namespace phi {

template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& label,
                      paddle::optional<const DenseTensor&> weight,
                      int64_t ignore_index,
                      const std::string& reduction,
                      DenseTensor* out,
                      DenseTensor* total_weight) {
  auto* x = &input;
  auto x_data = x->data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto total_weight_data = dev_ctx.template Alloc<T>(total_weight);
  auto label_data = label.data<int64_t>();
  auto weight_data = weight.get_ptr() ? weight.get_ptr()->data<T>() : nullptr;
#ifdef PADDLE_WITH_HIP
  hipMemset(total_weight_data, 0, sizeof(T));
#else
  cudaMemset(total_weight_data, 0, sizeof(T));
#endif
  auto x_dims = x->dims();
  auto batch_size = x_dims[0];
  auto n_classes = x_dims[1];
  int64_t size_average = (int64_t)(reduction == "mean");

  if (x_dims.size() == 2) {
    int blocks = NumBlocks(batch_size);
    int threads = kNumCUDAThreads;
    if (reduction == "none") {
      GPUNLLLossForward1D_no_reduce<
          T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_data,
                                                       x_data,
                                                       label_data,
                                                       weight_data,
                                                       batch_size,
                                                       n_classes,
                                                       ignore_index);
    } else {
      GPUNLLLossForward1D_with_reduce<T><<<1, NTHREADS, 0, dev_ctx.stream()>>>(
          out_data,
          total_weight_data,
          x_data,
          label_data,
          weight_data,
          batch_size,
          n_classes,
          size_average,
          ignore_index);
    }
  } else if (x_dims.size() == 4) {
    const auto in_dim2 = x_dims[2];
    const auto in_dim3 = x_dims[3];
    const auto map_size = in_dim2 * in_dim3;
    const auto out_numel = batch_size * in_dim2 * in_dim3;
    int blocks = NumBlocks(out_numel);
    int threads = kNumCUDAThreads;
    if (reduction == "none") {
      GPUNLLLossForward2D_no_reduce<
          T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_data,
                                                       x_data,
                                                       label_data,
                                                       weight_data,
                                                       batch_size,
                                                       n_classes,
                                                       in_dim2,
                                                       in_dim3,
                                                       ignore_index);
    } else {
      int blocks_per_sample = NumBlocks(map_size) / 128;
      blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
      int total_blocks = blocks_per_sample * batch_size;
      GPUNLLLossForward2D_with_reduce<
          T><<<total_blocks, threads, 0, dev_ctx.stream()>>>(out_data,
                                                             total_weight_data,
                                                             x_data,
                                                             label_data,
                                                             weight_data,
                                                             batch_size,
                                                             n_classes,
                                                             map_size,
                                                             blocks_per_sample,
                                                             ignore_index);
      if (size_average) {
        GPUNLLLossForward2D_size_average<T><<<1, 1, 0, dev_ctx.stream()>>>(
            out_data, total_weight_data);
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    nll_loss, GPU, ALL_LAYOUT, phi::NllLossRawKernel, float, double) {}
