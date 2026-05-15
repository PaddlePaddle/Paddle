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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/nll_loss.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& label,
                      const optional<DenseTensor>& weight,
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
  auto x_dims = x->dims();
  auto batch_size = x_dims[0];
  auto n_classes = x_dims[1];
  int size_average = static_cast<int>(reduction == "mean");
  using AccT = typename MPTypeTrait<T>::Type;
  if (x_dims.size() == 2) {
    int64_t blocks = NumBlocks(batch_size);
    int threads = kNumCUDAThreads;
    if (reduction == "none") {
#ifdef PADDLE_WITH_HIP
      hipMemsetAsync(total_weight_data, 0, sizeof(T), dev_ctx.stream());
#else
      cudaMemsetAsync(total_weight_data, 0, sizeof(T), dev_ctx.stream());
#endif
      GPUNLLLossForward1D_no_reduce<T>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(out_data,
                                                     x_data,
                                                     label_data,
                                                     weight_data,
                                                     batch_size,
                                                     n_classes,
                                                     ignore_index);
    } else {
      if (FLAGS_use_accuracy_compatible_kernel) {
        int nthreads = nll_loss_threads(batch_size);
        GPUNLLLossForward1D_with_reduce_compatible<T, AccT>
            <<<1, nthreads, nthreads * sizeof(AccT) * 2, dev_ctx.stream()>>>(
                out_data,
                total_weight_data,
                x_data,
                label_data,
                weight_data,
                batch_size,
                n_classes,
                size_average,
                ignore_index);
      } else {
        GPUNLLLossForward1D_with_reduce<T, AccT>
            <<<1, NTHREADS, 0, dev_ctx.stream()>>>(out_data,
                                                   total_weight_data,
                                                   x_data,
                                                   label_data,
                                                   weight_data,
                                                   batch_size,
                                                   n_classes,
                                                   size_average,
                                                   ignore_index);
      }
    }
  } else if (x_dims.size() == 4) {
    const auto in_dim2 = x_dims[2];
    const auto in_dim3 = x_dims[3];
    const auto map_size = in_dim2 * in_dim3;
    const auto out_numel = batch_size * in_dim2 * in_dim3;
    int64_t blocks = NumBlocks(out_numel);
    int threads = kNumCUDAThreads;
    if (reduction == "none") {
#ifdef PADDLE_WITH_HIP
      hipMemsetAsync(total_weight_data, 0, sizeof(T), dev_ctx.stream());
#else
      cudaMemsetAsync(total_weight_data, 0, sizeof(T), dev_ctx.stream());
#endif
      GPUNLLLossForward2D_no_reduce<T>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(out_data,
                                                     x_data,
                                                     label_data,
                                                     weight_data,
                                                     batch_size,
                                                     n_classes,
                                                     in_dim2,
                                                     in_dim3,
                                                     ignore_index);
    } else {
      // Use kNumCUDAThreads4D=1024 to match PyTorch's CUDA_NUM_THREADS in
      // NLLLoss2d.cu, and compute blocks_per_sample the same way PyTorch does:
      //   GET_BLOCKS(map_nelem) / 128  where GET_BLOCKS uses 1024 threads.
      int threads4d = kNumCUDAThreads4D;
      int blocks_per_sample =
          static_cast<int>((map_size + kNumCUDAThreads4D - 1) /
                           kNumCUDAThreads4D) /
          128;
      blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
      int64_t total_blocks = blocks_per_sample * batch_size;
#ifdef PADDLE_WITH_HIP
      hipMemsetAsync(out_data, 0, sizeof(T), dev_ctx.stream());
      hipMemsetAsync(total_weight_data, 0, sizeof(T), dev_ctx.stream());
#else
      cudaMemsetAsync(out_data, 0, sizeof(T), dev_ctx.stream());
      cudaMemsetAsync(total_weight_data, 0, sizeof(T), dev_ctx.stream());
#endif
      // Both compatible and non-compatible paths use PyTorch NLLLoss2d.cu
      // topology: multi-block per-sample traversal (blocks_per_sample blocks
      // per sample), BlockReduceSum within each block, gpuAtomicAdd into
      // global output/total_weight, and a separate single-thread mean kernel.
      // The previous compatible path used a single-block butterfly reduction
      // over the full batch*map_nelem which diverges from PyTorch accumulation
      // order and is the dominant source of accuracy_error for rank>2 inputs.
      GPUNLLLossForward2D_with_reduce<T, AccT>
          <<<total_blocks, threads4d, 0, dev_ctx.stream()>>>(out_data,
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
        GPUNLLLossForward2D_size_average<T>
            <<<1, 1, 0, dev_ctx.stream()>>>(out_data, total_weight_data);
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    nll_loss, GPU, ALL_LAYOUT, phi::NllLossRawKernel, float, double) {}
