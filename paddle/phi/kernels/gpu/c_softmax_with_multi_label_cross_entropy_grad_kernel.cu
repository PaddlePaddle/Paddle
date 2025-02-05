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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/utils/string/string_helper.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int64_t kNumMaximumNumBlocks = 4096;

static inline int64_t NumBlocks(const int64_t N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void CalculateSoftLogitsGrad(T* logits_grad,
                                        const IndexT* labels,
                                        const T* smooth_weight,
                                        const IndexT ignore_index,
                                        const int64_t start_index,
                                        const int64_t end_index,
                                        const int64_t N,
                                        const int64_t D,
                                        const int64_t C) {
  CUDA_KERNEL_LOOP_TYPE(i, N, int64_t) {
    for (int j = 0; j < C; ++j) {
      auto real_label = labels[i * C + j];
      auto prob = smooth_weight[i * C + j];
      if (real_label >= start_index && real_label < end_index) {
        int64_t idx = i * D + real_label - start_index;
        logits_grad[idx] = real_label == ignore_index
                               ? static_cast<T>(0)
                               : (logits_grad[idx] - prob);
      }
    }
  }
}

template <typename T, typename IndexT>
__global__ void SoftMaskLabelByIndexGrad(T* logits_grad,
                                         const T* loss_grad,
                                         const int64_t N,
                                         const int64_t D,
                                         const int64_t C,
                                         const bool sum_multi_label_loss) {
  CUDA_KERNEL_LOOP_TYPE(i, N * D, int64_t) {
    auto loss_grad_idx = sum_multi_label_loss ? (i / D) : (i / D * C);
    logits_grad[i] *= loss_grad[loss_grad_idx];
  }
}

template <typename T, typename Context>
void CSoftmaxWithMultiLabelCrossEntropyGradKernel(
    const Context& dev_ctx,
    const DenseTensor& softmax_in,
    const DenseTensor& label_in,
    const DenseTensor& smooth_weight_in,
    const DenseTensor& loss_grad_in,
    int64_t ignore_index,
    bool sum_multi_label_loss,
    int rank,
    int nranks,
    DenseTensor* logits_grad) {
  const phi::DenseTensor* labels = &label_in;
  const phi::DenseTensor* smooth_weight = &smooth_weight_in;
  const phi::DenseTensor* loss_grad = &loss_grad_in;
  const phi::DenseTensor* softmax = &softmax_in;
  phi::DenseTensor* logit_grad = logits_grad;

  if (logit_grad != softmax) {
    phi::Copy(dev_ctx, *softmax, dev_ctx.GetPlace(), false, logit_grad);
  }
  const auto softmax_dims = softmax->dims();
  const int axis = softmax_dims.size() - 1;
  const int64_t N = phi::funcs::SizeToAxis<int64_t>(axis, softmax_dims);
  const int64_t D = phi::funcs::SizeFromAxis<int64_t>(axis, softmax_dims);

  const auto label_dims = labels->dims();
  const int64_t C = label_dims[axis];

  phi::DenseTensor logit_grad_2d;
  logit_grad_2d.ShareDataWith(*logit_grad).Resize({N, D});

  int64_t blocks = NumBlocks(N * D);
  int64_t blocks_cal = NumBlocks(N);
  int threads = kNumCUDAThreads;
  const auto& label_type = labels->dtype();
  const int64_t start_index = rank * D;
  const int64_t end_index = start_index + D;

  if (label_type == phi::DataType::INT32) {
    CalculateSoftLogitsGrad<T, int32_t>
        <<<blocks_cal, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                       labels->data<int32_t>(),
                                                       smooth_weight->data<T>(),
                                                       ignore_index,
                                                       start_index,
                                                       end_index,
                                                       N,
                                                       D,
                                                       C);

    SoftMaskLabelByIndexGrad<T, int32_t>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                   loss_grad->data<T>(),
                                                   N,
                                                   D,
                                                   C,
                                                   sum_multi_label_loss);
  } else if (label_type == phi::DataType::INT64) {
    CalculateSoftLogitsGrad<T, int64_t>
        <<<blocks_cal, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                       labels->data<int64_t>(),
                                                       smooth_weight->data<T>(),
                                                       ignore_index,
                                                       start_index,
                                                       end_index,
                                                       N,
                                                       D,
                                                       C);

    SoftMaskLabelByIndexGrad<T, int64_t>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                   loss_grad->data<T>(),
                                                   N,
                                                   D,
                                                   C,
                                                   sum_multi_label_loss);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(c_softmax_with_multi_label_cross_entropy_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CSoftmaxWithMultiLabelCrossEntropyGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
