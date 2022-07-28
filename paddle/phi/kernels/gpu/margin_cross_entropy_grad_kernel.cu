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

// old op include, fluid should be removed
#ifdef PADDLE_WITH_HIP
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <vector>
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/impl/softmax_kernel_impl.h"
#include "paddle/phi/kernels/margin_cross_entropy_grad_kernel.h"
#include "paddle/phi/kernels/margin_cross_entropy_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
// trace op include
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi {
using Tensor = DenseTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T, typename Context>
void GetClassInterval(const gpuStream_t& stream,
                      const phi::Place& place,
                      const Context& ctx,
                      const int rid,
                      const int rank,
                      const int nranks,
                      const int D,
                      Tensor* class_interval) {
  std::vector<int> shard_dim_vec(nranks + 1, 0);
  shard_dim_vec[rank + 1] = D;
  if (nranks <= 1) {
    paddle::framework::TensorFromVector(shard_dim_vec, ctx, class_interval);
    return;
  }
}
template <typename T, typename IndexT>
__global__ void CalculateGrad(T* logits_grad,
                              const T* loss_grad,
                              const T* logits,
                              const IndexT* label,
                              const float margin1,
                              const float margin2,
                              const float scale,
                              const int rank,
                              const int64_t N,
                              const int64_t D,
                              const int* class_interval_ptr) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  int start_index = class_interval_ptr[rank];
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == label[row]) {
      logits_grad[i] = (logits_grad[i] - static_cast<T>(1.0)) * loss_grad[row];
      if (fabs(margin1 - 1.0) > 1e-8 || fabs(margin2) > 1e-8) {
        MPType dout = static_cast<MPType>(logits_grad[i]);
        MPType one = static_cast<MPType>(1.0f);
        MPType x = static_cast<MPType>(logits[i]);
        MPType m1 = static_cast<MPType>(margin1);
        MPType m2 = static_cast<MPType>(margin2);

        MPType d = m1 * sin(m1 * acos(x) + m2) / sqrt(one - x * x);
        logits_grad[i] = static_cast<T>(dout * d);
      }
    } else {
      logits_grad[i] *= loss_grad[row];
    }
    if (fabs(scale - 1.0) > 1e-8) {
      logits_grad[i] *= static_cast<T>(scale);
    }
  }
}

template <typename T, typename Context>
void MarginCrossEntropyGradKernel(const Context& dev_ctx,
                                  const DenseTensor& logits,
                                  const DenseTensor& label,
                                  const DenseTensor& softmax,
                                  const DenseTensor& loss_grad,
                                  bool return_softmax,
                                  int ring_id,
                                  int rank,
                                  int nranks,
                                  float margin1,
                                  float margin2,
                                  float margin3,
                                  float scale,
                                  DenseTensor* logit_grad) {
  const auto softmax_dims = softmax.dims();
  const int axis = softmax_dims.size() - 1;
  const int N = phi::funcs::SizeToAxis(axis, softmax_dims);
  const int D = phi::funcs::SizeFromAxis(axis, softmax_dims);

  if (return_softmax) {
    phi::Copy(dev_ctx, softmax, dev_ctx.GetPlace(), false, logit_grad);
  } else {
    logit_grad->ShareDataWith(softmax);
  }

  int blocks = NumBlocks(N * D);
  int threads = kNumCUDAThreads;
  const auto& label_type =
      paddle::framework::TransToProtoVarType(label.dtype());

  Tensor class_interval;
  GetClassInterval<T, Context>(dev_ctx.stream(),
                               dev_ctx.GetPlace(),
                               dev_ctx,
                               ring_id,
                               rank,
                               nranks,
                               D,
                               &class_interval);

  if (label_type == paddle::framework::proto::VarType::INT32) {
    typedef int32_t LabelT;
    CalculateGrad<T, LabelT>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad->data<T>(),
                                                   loss_grad.data<T>(),
                                                   logits.data<T>(),
                                                   label.data<LabelT>(),
                                                   margin1,
                                                   margin2,
                                                   scale,
                                                   rank,
                                                   N,
                                                   D,
                                                   class_interval.data<int>());
  } else if (label_type == paddle::framework::proto::VarType::INT64) {
    typedef int64_t LabelT;
    CalculateGrad<T, LabelT>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad->data<T>(),
                                                   loss_grad.data<T>(),
                                                   logits.data<T>(),
                                                   label.data<LabelT>(),
                                                   margin1,
                                                   margin2,
                                                   scale,
                                                   rank,
                                                   N,
                                                   D,
                                                   class_interval.data<int>());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(margin_cross_entropy_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MarginCrossEntropyGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
