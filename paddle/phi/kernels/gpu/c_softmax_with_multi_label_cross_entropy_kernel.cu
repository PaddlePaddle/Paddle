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
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/utils/string/string_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename Context, typename T>
struct CSoftmaxWithMultiLabelCrossEntropyFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& logits,
                  const DenseTensor& label,
                  const DenseTensor& smooth_weight,
                  int64_t ignore_index,
                  bool sum_multi_label_loss,
                  int rank,
                  int nranks,
                  DenseTensor* softmax,
                  DenseTensor* loss);
};

static constexpr int kNumCUDAThreads = 512;
static constexpr int64_t kNumMaximumNumBlocks = 4096;

static inline int64_t NumBlocks(const int64_t N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void SoftMaskLabelByIndex(T* predicted_logits,
                                     const T* logit,
                                     const IndexT* label,
                                     const IndexT ignore_index,
                                     const int64_t start_index,
                                     const int64_t end_index,
                                     const int64_t N,
                                     const int64_t D,
                                     const int64_t C,
                                     const int nranks) {
  CUDA_KERNEL_LOOP_TYPE(i, N, int64_t) {
    for (int j = 0; j < C; ++j) {
      auto real_label = label[i * C + j];
      PADDLE_ENFORCE(((real_label < D * nranks) && (real_label >= 0)) ||
                         (real_label == ignore_index),
                     "The index is out of bounds, "
                     "please check whether the value of label and "
                     "input meet the class number. It should "
                     "be less than [%ld] or equal to [%ld], but received [%ld]",
                     static_cast<int64_t>(D * nranks),
                     static_cast<int64_t>(ignore_index),
                     static_cast<int64_t>(real_label));

      if (real_label >= start_index && real_label < end_index) {
        predicted_logits[i * C + j] = logit[i * D + real_label - start_index];
      }
    }
  }
}

template <typename T, typename IndexT>
__global__ void CalculateSoftLoss(T* loss,
                                  const T* predict_logits,
                                  const T* sum_exp_logits,
                                  const IndexT* label,
                                  const T* smooth_weight,
                                  const int64_t ignore_index,
                                  const int64_t N,
                                  const int64_t C,
                                  const bool sum_multi_label_loss) {
  CUDA_KERNEL_LOOP_TYPE(i, N, int64_t) {
    T tmp_loss = static_cast<T>(0);
    loss[i] = static_cast<T>(0);
    T log_sum_exp_logits = phi::funcs::TolerableValue<T>()(
        phi::funcs::real_log(sum_exp_logits[i]));
    for (int j = 0; j < C; ++j) {
      int64_t label_idx = i * C + j;
      auto real_label = static_cast<int64_t>(label[label_idx]);
      auto prob = static_cast<T>(smooth_weight[label_idx]);
      tmp_loss =
          ignore_index == real_label
              ? static_cast<T>(0)
              : phi::funcs::TolerableValue<T>()(
                    (log_sum_exp_logits - predict_logits[label_idx]) * prob);
      if (sum_multi_label_loss) {
        loss[i] += tmp_loss;
      } else {
        loss[label_idx] = tmp_loss;
      }
    }
  }
}

template <typename T, typename Context>
void CSoftmaxWithMultiLabelCrossEntropyKernel(const Context& dev_ctx,
                                              const DenseTensor& logits,
                                              const DenseTensor& label,
                                              const DenseTensor& smooth_weight,
                                              int64_t ignore_index,
                                              bool sum_multi_label_loss,
                                              int rank,
                                              int nranks,
                                              DenseTensor* softmax,
                                              DenseTensor* loss) {
  CSoftmaxWithMultiLabelCrossEntropyFunctor<phi::GPUContext, T> functor_;
  functor_(dev_ctx,
           logits,
           label,
           smooth_weight,
           ignore_index,
           sum_multi_label_loss,
           rank,
           nranks,
           softmax,
           loss);
}

template <typename T>
struct CSoftmaxWithMultiLabelCrossEntropyFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& logits_in,
                  const DenseTensor& label_in,
                  const DenseTensor& smooth_weight_in,
                  int64_t ignore_index,
                  bool sum_multi_label_loss,
                  int rank,
                  int nranks,
                  DenseTensor* softmax,
                  DenseTensor* loss) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    const phi::DenseTensor* logits = &logits_in;
    const phi::DenseTensor* labels = &label_in;
    const phi::DenseTensor* smooth_weight = &smooth_weight_in;

    gpuStream_t stream = nullptr;
    phi::distributed::NCCLCommContext* comm_ctx = nullptr;

    comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
        dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

    stream = dev_ctx.stream();

    // allocate memory on device.
    dev_ctx.template Alloc<T>(softmax);
    dev_ctx.template Alloc<T>(loss);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int64_t N = phi::funcs::SizeToAxis<int64_t>(axis, logits_dims);
    const int64_t D = phi::funcs::SizeFromAxis<int64_t>(axis, logits_dims);
    const int64_t C = phi::funcs::SizeFromAxis<int64_t>(axis, labels_dims);

    phi::DenseTensor logits_2d, softmax_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({N, D});
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    int64_t loss_last_dim = sum_multi_label_loss ? 1 : C;
    loss_2d.ShareDataWith(*loss).Resize({N, loss_last_dim});

    auto eigen_logits = phi::funcs::EigenMatrix<T>::From(logits_2d);
    auto eigen_softmax = phi::funcs::EigenMatrix<T>::From(softmax_2d);

    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max.Resize({N, 1});
    dev_ctx.template Alloc<T>(&logits_max);

    auto eigen_logits_max = phi::funcs::EigenMatrix<T>::From(logits_max);
    Eigen::DSizes<int, 1> along_axis(1);
    eigen_logits_max.device(*dev_ctx.eigen_device()) =
        eigen_logits.maximum(along_axis);

    comm_ctx->AllReduce(&logits_max, logits_max, ncclMax, stream);

    // step 2, obtain logit - logit_max
    Eigen::DSizes<int, 2> batch_by_one(N, 1);
    Eigen::DSizes<int, 2> one_by_class(1, D);

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_logits -
         eigen_logits_max.reshape(batch_by_one).broadcast(one_by_class));

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits.Resize({N, C});
    dev_ctx.template Alloc<T>(&predicted_logits);

    auto t = phi::EigenVector<T>::Flatten(predicted_logits);
    t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

    const int64_t start_index = rank * D;
    const int64_t end_index = start_index + D;

    int64_t blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = labels->dtype();

    if (label_type == phi::DataType::INT32) {
      SoftMaskLabelByIndex<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(
              predicted_logits.data<T>(),
              softmax_2d.data<T>(),
              labels->data<int32_t>(),
              static_cast<int32_t>(ignore_index),
              start_index,
              end_index,
              N,
              D,
              C,
              nranks);
    } else if (label_type == phi::DataType::INT64) {
      SoftMaskLabelByIndex<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(
              predicted_logits.data<T>(),
              softmax_2d.data<T>(),
              labels->data<int64_t>(),
              static_cast<int32_t>(ignore_index),
              start_index,
              end_index,
              N,
              D,
              C,
              nranks);
    }

    dev_ctx.template Alloc<T>(&predicted_logits);
    comm_ctx->AllReduce(&predicted_logits, predicted_logits, ncclSum, stream);

    // step 4, obtain exp(logit)
    eigen_softmax.device(*dev_ctx.eigen_device()) = eigen_softmax.exp();

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits.Resize({N, 1});
    dev_ctx.template Alloc<T>(&sum_exp_logits);

    phi::SumKernel<T, phi::GPUContext>(
        dev_ctx, softmax_2d, {-1}, softmax_2d.dtype(), true, &sum_exp_logits);

    comm_ctx->AllReduce(&sum_exp_logits, sum_exp_logits, ncclSum, stream);

    if (label_type == phi::DataType::INT32) {
      CalculateSoftLoss<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_2d.data<T>(),
                                                     predicted_logits.data<T>(),
                                                     sum_exp_logits.data<T>(),
                                                     labels->data<int32_t>(),
                                                     smooth_weight->data<T>(),
                                                     ignore_index,
                                                     N,
                                                     C,
                                                     sum_multi_label_loss);

    } else {
      CalculateSoftLoss<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_2d.data<T>(),
                                                     predicted_logits.data<T>(),
                                                     sum_exp_logits.data<T>(),
                                                     labels->data<int64_t>(),
                                                     smooth_weight->data<T>(),
                                                     ignore_index,
                                                     N,
                                                     C,
                                                     sum_multi_label_loss);
    }

    auto eigen_sum_exp_logits =
        phi::funcs::EigenMatrix<T>::From(sum_exp_logits);
    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax *
         eigen_sum_exp_logits.inverse().broadcast(one_by_class));
#endif
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(c_softmax_with_multi_label_cross_entropy,
                   GPU,
                   ALL_LAYOUT,
                   phi::CSoftmaxWithMultiLabelCrossEntropyKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
