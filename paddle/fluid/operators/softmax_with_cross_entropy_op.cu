/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <cub/cub.cuh>
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

namespace {
template <typename T>
__global__ void CrossEntropyGrad(T* logit_grad, const int64_t* labels,
                                 const int batch_size, const int class_num,
                                 const int ignore_index) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size;
       i += blockDim.x * gridDim.x) {
    int idx = i * class_num + labels[i];
    logit_grad[idx] -=
        ignore_index == labels[i] ? static_cast<T>(0.) : static_cast<T>(1.);
  }
}

template <typename T>
__global__ void Scale(T* logit_grad, const T* loss_grad, const int num,
                      const int class_num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    logit_grad[i] *= loss_grad[i / class_num];
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels,
                                               const int batch_size,
                                               const int class_num) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < batch_size * class_num) {
    int row_ids = ids / class_num;
    logit_grad[ids] = loss_grad[row_ids] * (logit_grad[ids] - labels[ids]);
  }
}

}  // namespace

static __device__ __forceinline__ platform::float16 exp_on_device(
    platform::float16 x) {
  return ::Eigen::numext::exp(x);
}
static __device__ __forceinline__ float exp_on_device(float x) {
  return expf(x);
}
static __device__ __forceinline__ double exp_on_device(double x) {
  return exp(x);
}
static __device__ __forceinline__ platform::float16 log_on_device(
    platform::float16 x) {
  return math::TolerableValue<platform::float16>()(::Eigen::numext::log(x));
}
static __device__ __forceinline__ float log_on_device(float x) {
  return math::TolerableValue<float>()(logf(x));
}
static __device__ __forceinline__ double log_on_device(double x) {
  return math::TolerableValue<double>()(log(x));
}

/** In the following codes, 3 CUDA kernels are implemented to calculate softmax
 * and loss **/
/*
  Supposing the x is `logits` and y is `labels`, the equations are as
followings:
  cross\_entropy_i = \sum_{j}[- y_i_j * log({e^{x_i_j}/\sum_{j}e^{x_i_j}})]
        = \sum_{j}[- y_i_j * log({e^{x_i_j - max_i}/\sum_{j}e^{x_i_j-max_i}})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - log\sum_{j}e^{x_i_j - max_i})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - logDiffMaxSum_i)]
        = \sum_{j}(-y_i_j * tmp_i_j)
  softmax_i_j = e^{tmp_i_j}
where:
  max_i = \max_{j}{x_i_j}
  logDiffMaxSum_i = log\sum_{j}e^{x_i_j - max_i}
  tmp_i_j = x_i_j - max_i - logDiffMaxSum_i
Therefore, the calculation can be separated into 3 steps:
Step 1: row-wise operation to calculate max_i
Step 2: row-wise operation to calculate logDiffMaxSum_i
Step 3: caculate tmp_i_j, and finally get softmax_i_j and cross\_entropy_i
To save memory, we can share memory among max_i, logDiffMaxSum_i and
cross\_entropy_i.
In this way, the 3 steps should be changed to:
Step 1 (RowReductionForMax): row-wise operation to calculate max_i
Step 2 (RowReductionForDiffMaxSum): calculate immediate result of softmax'_i_j =
x_i_j - max_i, and row-wise operation to calculate logDiffMaxSum_i
Step 3 (RowReductionForSoftmaxAndCrossEntropy): calculate tmp_i_j = softmax'_i_j
- logDiffMaxSum_i, and finally get softmax_i_j and cross\_entropy_i
*/

// There are 3 kinds of reduce algorithms in cub:
// BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
// BLOCK_REDUCE_RAKING
// BLOCK_REDUCE_WARP_REDUCTIONS (default)
template <typename T, int BlockDim>
using BlockReduce =
    cub::BlockReduce<T, BlockDim /*, cub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

// Make sure that BlockDim <= feature_size
// This kernel is used to calculate the max element of each row
template <typename T, int BlockDim>
static __global__ void RowReductionForMax(const T* logits_data, T* max_data,
                                          int feature_size) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  auto beg_idx = feature_size * blockIdx.x + threadIdx.x;
  auto end_idx = feature_size * (blockIdx.x + 1);

  T cur_max = logits_data[beg_idx];
  beg_idx += BlockDim;
  while (beg_idx < end_idx) {
    if (cur_max < logits_data[beg_idx]) {
      cur_max = logits_data[beg_idx];
    }
    beg_idx += BlockDim;
  }

  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, cub::Max());

  if (threadIdx.x == 0) {
    max_data[blockIdx.x] =
        cur_max < static_cast<T>(-64) ? static_cast<T>(-64) : cur_max;
  }
}

// Make sure that BlockDim <= feature_size
template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiffMaxSum(const T* logits_data,
                                                 T* max_data, T* softmax,
                                                 int feature_size) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  auto beg_idx = feature_size * blockIdx.x + threadIdx.x;
  auto end_idx = feature_size * (blockIdx.x + 1);

  auto block_max = max_data[blockIdx.x];

  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  auto idx = beg_idx + BlockDim;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    idx += BlockDim;
  }

  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
  if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);

  if (!CalculateLogSoftmax) return;
  __syncthreads();
  diff_max_sum = max_data[blockIdx.x];
  softmax[beg_idx] -= diff_max_sum;
  beg_idx += BlockDim;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    beg_idx += BlockDim;
  }
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
}

// Make sure that BlockDim <= feature_size
template <typename T, int BlockDim>
static __global__ void RowReductionForSoftmaxAndCrossEntropy(
    const T* logits_data, const T* labels_data, T* loss_data, T* softmax,
    int feature_size) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  auto beg_idx = feature_size * blockIdx.x + threadIdx.x;
  auto end_idx = feature_size * (blockIdx.x + 1);

  // log_diff_max_sum shares memory with loss
  auto block_log_diff_max_sum = loss_data[blockIdx.x];
  auto tmp = softmax[beg_idx] - block_log_diff_max_sum;
  softmax[beg_idx] = exp_on_device(tmp);
  auto loss = -labels_data[beg_idx] * tmp;
  beg_idx += BlockDim;
  while (beg_idx < end_idx) {
    tmp = softmax[beg_idx] - block_log_diff_max_sum;
    softmax[beg_idx] = exp_on_device(tmp);
    loss -= (labels_data[beg_idx] * tmp);
    beg_idx += BlockDim;
  }

  loss = BlockReduce<T, BlockDim>(temp_storage).Reduce(loss, cub::Sum());
  if (threadIdx.x == 0) loss_data[blockIdx.x] = loss;
}

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctor {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctor(const T* logits,
                                          const int64_t* labels, T* loss,
                                          T* log_softmax, int feature_size)
      : logits_(logits),
        labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        feature_size_(feature_size) {}

  __device__ void operator()(int idx) const {
    auto row_idx = idx / feature_size_;
    auto col_idx = idx % feature_size_;
    if (col_idx != labels_[row_idx]) {
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
    } else {
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[row_idx] = -softmax;
    }
  }

 private:
  const T* logits_;
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int feature_size_;
};

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx(const T* logits,
                                                       const int64_t* labels,
                                                       T* loss, T* log_softmax,
                                                       int feature_size,
                                                       int ignore_idx)
      : logits_(logits),
        labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        feature_size_(feature_size),
        ignore_idx_(ignore_idx) {}

  __device__ void operator()(int idx) const {
    auto row_idx = idx / feature_size_;
    auto col_idx = idx % feature_size_;
    if (col_idx != labels_[row_idx] || col_idx == ignore_idx_) {
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
    } else {
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[row_idx] = -softmax;
    }
  }

 private:
  const T* logits_;
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int feature_size_;
  int ignore_idx_;
};

template <typename T>
static __global__ void SetSoftmaxToOneWhenFeatureSizeIsOne(T* out,
                                                           int batch_size) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) out[idx] = static_cast<T>(1);
}

template <typename T>
static void HardLabelSoftmaxWithCrossEntropy(
    const platform::CUDADeviceContext& ctx, const T* logits_data,
    const int64_t* labels_data, T* loss_data, T* softmax_data, int batch_size,
    int feature_size, int ignore_idx) {
  constexpr int kMaxBlockDim = 512;
  int block_dim = feature_size >= kMaxBlockDim
                      ? kMaxBlockDim
                      : (1 << static_cast<int>(std::log2(feature_size)));
  auto stream = ctx.stream();

#define CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)    \
  case BlockDim: {                                                           \
    RowReductionForMax<T, BlockDim><<<batch_size, BlockDim, 0, stream>>>(    \
        logits_data, loss_data, feature_size);                               \
    RowReductionForDiffMaxSum<T, BlockDim,                                   \
                              true><<<batch_size, BlockDim, 0, stream>>>(    \
        logits_data, loss_data, softmax_data, feature_size);                 \
    platform::ForRange<platform::CUDADeviceContext> for_range(               \
        ctx, batch_size* feature_size);                                      \
    if (ignore_idx >= 0 && ignore_idx < feature_size) {                      \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx<T>(     \
          logits_data, labels_data, loss_data, softmax_data, feature_size,   \
          ignore_idx));                                                      \
    } else {                                                                 \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctor<T>(                  \
          logits_data, labels_data, loss_data, softmax_data, feature_size)); \
    }                                                                        \
  } break

  switch (block_dim) {
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    case 1:
      SetSoftmaxToOneWhenFeatureSizeIsOne<<<(batch_size + kMaxBlockDim - 1) /
                                                kMaxBlockDim,
                                            kMaxBlockDim, 0, stream>>>(
          softmax_data, batch_size);
      cudaMemsetAsync(loss_data, 0, batch_size * sizeof(T), stream);
      break;
    default:
      PADDLE_THROW("BlockDim must be 2^n in softmax_with_cross_entropy_op");
      break;
  }
#undef CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

template <typename T>
static void SoftmaxWithCrossEntropyFusedKernel(const T* logits_data,
                                               const T* labels_data,
                                               T* softmax_data, T* loss_data,
                                               int batch_size, int feature_size,
                                               cudaStream_t stream) {
  constexpr int kMaxBlockDim = 512;
  int block_dim = feature_size >= kMaxBlockDim
                      ? kMaxBlockDim
                      : (1 << static_cast<int>(std::log2(feature_size)));

#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                \
  case BlockDim:                                                              \
    RowReductionForMax<T, BlockDim><<<batch_size, BlockDim, 0, stream>>>(     \
        logits_data, loss_data, feature_size);                                \
    RowReductionForDiffMaxSum<T,                                              \
                              BlockDim><<<batch_size, BlockDim, 0, stream>>>( \
        logits_data, loss_data, softmax_data, feature_size);                  \
    RowReductionForSoftmaxAndCrossEntropy<                                    \
        T, BlockDim><<<batch_size, BlockDim, 0, stream>>>(                    \
        logits_data, labels_data, loss_data, softmax_data, feature_size);     \
    break

  switch (block_dim) {
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    case 1:
      SetSoftmaxToOneWhenFeatureSizeIsOne<<<(batch_size + kMaxBlockDim - 1) /
                                                kMaxBlockDim,
                                            kMaxBlockDim, 0, stream>>>(
          softmax_data, batch_size);
      cudaMemsetAsync(loss_data, 0, batch_size * sizeof(T), stream);
      break;
    default:
      PADDLE_THROW("BlockDim must be 2^n in softmax_with_cross_entropy_op");
      break;
  }

#undef CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");

    Tensor* loss = context.Output<Tensor>("Loss");
    auto* softmax_data = softmax->mutable_data<T>(context.GetPlace());
    auto* loss_data = loss->mutable_data<T>(context.GetPlace());

    auto soft_label = context.Attr<bool>("soft_label");
    auto ignore_index = context.Attr<int>("ignore_index");

    int rank = logits->dims().size();
    if (soft_label) {
      int batch_size = 1;
      for (int i = 0; i < rank - 1; ++i) {
        batch_size *= logits->dims()[i];
      }

      int feature_size = logits->dims()[rank - 1];
      auto* logits_data = logits->data<T>();
      auto* labels_data = labels->data<T>();
      SoftmaxWithCrossEntropyFusedKernel(
          logits_data, labels_data, softmax_data, loss_data, batch_size,
          feature_size, context.cuda_device_context().stream());
    } else {
      if (!context.Attr<bool>("numeric_stable_mode")) {
        // reshape to 2d
        Tensor logits_2d = framework::ReshapeToMatrix(*logits, rank - 1);
        Tensor softmax_2d = framework::ReshapeToMatrix(*softmax, rank - 1);
        Tensor loss_2d = framework::ReshapeToMatrix(*loss, rank - 1);
        Tensor labels_2d = framework::ReshapeToMatrix(*labels, rank - 1);

        math::SoftmaxCUDNNFunctor<T>()(context.cuda_device_context(),
                                       &logits_2d, &softmax_2d);
        math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
            context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
            false, ignore_index);
      } else {
        int batch_size = 1;
        for (int i = 0; i < rank - 1; ++i) {
          batch_size *= logits->dims()[i];
        }
        int feature_size = logits->dims()[rank - 1];
        auto* logits_data = logits->data<T>();
        auto* labels_data = labels->data<int64_t>();
        HardLabelSoftmaxWithCrossEntropy<T>(
            context.cuda_device_context(), logits_data, labels_data, loss_data,
            softmax_data, batch_size, feature_size, ignore_index);
      }
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");
    const Tensor* labels = context.Input<Tensor>("Label");
    const T* loss_grad_data =
        context.Input<Tensor>(framework::GradVarName("Loss"))->data<T>();
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    framework::TensorCopy(*context.Input<Tensor>("Softmax"), context.GetPlace(),
                          context.device_context(), logit_grad);
    T* logit_grad_data = logit_grad->data<T>();

    int rank = logit_grad->dims().size();
    int batch_size = 1;
    for (int i = 0; i < rank - 1; ++i) {
      batch_size *= logit_grad->dims()[i];
    }

    const int class_num = logit_grad->dims()[rank - 1];
    int block = 512;
    auto stream = context.cuda_device_context().stream();
    auto ignore_index = context.Attr<int>("ignore_index");
    if (context.Attr<bool>("soft_label")) {
      int grid = (batch_size * class_num + block - 1) / block;
      const T* label_data = labels->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, batch_size, class_num);
    } else {
      int grid = (batch_size + block - 1) / block;
      const int64_t* label_data = labels->data<int64_t>();
      CrossEntropyGrad<T><<<grid, block, 0, stream>>>(
          logit_grad_data, label_data, batch_size, class_num, ignore_index);
      int num = batch_size * class_num;
      grid = (num + block - 1) / block;
      Scale<T><<<grid, block, 0, stream>>>(logit_grad_data, loss_grad_data, num,
                                           class_num);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<double>);
