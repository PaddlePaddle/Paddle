/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/thrust_iterator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

namespace {
template <typename T>
__global__ void CrossEntropyGrad(T* logit_grad, const int64_t* labels,
                                 const int batch_size, const int class_num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size;
       i += blockDim.x * gridDim.x) {
    int idx = i * class_num + labels[i];
    logit_grad[idx] -= static_cast<T>(1.);
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

__device__ inline float real_exp(float x) { return __expf(x); }

__device__ inline double real_exp(double x) { return exp(x); }

__device__ inline float real_log(float x) {
  // return logf(x);
  return math::TolerableValue<float>()(__logf(x));
}

__device__ inline double real_log(double x) {
  // return log(x);
  return math::TolerableValue<double>()(log(x));
}

template <typename T>
struct ValueClipMaxFunctor {
  __host__ __device__ inline T operator()(T x, T y) {
    auto tmp = (x < y ? y : x);
    return tmp < static_cast<T>(64) ? static_cast<T>(64) : tmp;
  }
};

template <typename T>
struct DiffExpFunctor {
  template <typename Tuple>
  __host__ __device__ T operator()(Tuple&& tuple) const {
    auto logit = thrust::get<0>(tuple);
    auto max = thrust::get<1>(tuple);
    return real_exp(logit - max);
  }
};

/*
template <typename T>
struct DiffExpFunctor2 {
  T max_;
  __host__ __device__ DiffExpFunctor2(T max) : max_(max) {}
  __host__ __device__ T operator()(T logit) const {
    return real_exp(logit - max_);
  }
};
*/

template <typename T>
struct SoftmaxImmediateFunctor {
  template <typename Tuple>
  __host__ __device__ T operator()(Tuple&& tuple) const {
    auto logit = thrust::get<0>(tuple);
    // auto label = thrust::get<1>(tuple);
    auto max = thrust::get<1>(tuple);
    auto max_sum = thrust::get<2>(tuple);
    return real_exp(logit - max) / max_sum;
  }
};

template <typename T>
struct CrossEntropyImmediateFunctor {
  template <typename Tuple>
  __host__ __device__ T operator()(Tuple&& tuple) const {
    return -thrust::get<1>(tuple) * real_log(thrust::get<0>(tuple));
  }
};

/*
template <typename T>
__global__ void RowMaxReduce(const T *logits_data, T *max_data, int batch_size,
int feature_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < batch_size) {
    auto it = logits_data + idx * feature_size;
    max_data[idx] = thrust::reduce(thrust::device, it, it + feature_size,
static_cast<T>(64), thrust::maximum<T>());
  }
}

template <typename T>
__global__ void RowDiffExpReduce(const T *logits_data, const T *max_data, T
*diff_exp_sum, int batch_size, int feature_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < batch_size) {
    auto it = logits_data + idx * feature_size;
    diff_exp_sum[idx] = thrust::transform_reduce(thrust::device, it, it +
feature_size,
        DiffExpFunctor2<T>(max_data[idx]), static_cast<T>(0),
thrust::plus<T>()); // NOLINT
  }
}

template <typename T>
__global__ void SoftmaxImmediateKernel(const T *logits_data, const T *max_data,
const T *diff_exp_sum, T *softmax, int batch_size, int feature_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < batch_size * feature_size) {
    int row = idx / feature_size;
    softmax[idx] = real_exp(logits_data[idx] - max_data[row]) /
diff_exp_sum[row];
  }
}

template <typename T>
__global__ void CrossEntropyReduce(const T *labels, const T *softmax, T *loss,
int batch_size, int feature_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < batch_size) {
    auto it = platform::MakeZipIterator(softmax + idx * feature_size, labels +
idx * feature_size);
    loss[idx] = thrust::transform_reduce(thrust::device, it, it + feature_size,
        CrossEntropyImmediateFunctor<T>(), static_cast<T>(0),
thrust::plus<T>()); // NOLINT
  }
}
*/

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
    softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());

    if (!context.Attr<bool>("soft_label")) {
      math::SoftmaxFunctor<platform::CUDADeviceContext, T>()(
          context.cuda_device_context(), logits, softmax);
      math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
          context.cuda_device_context(), loss, softmax, labels, false);
    } else {
      auto place = context.GetPlace();
      Tensor diff_exp_sum;
      int batch_size = labels->dims()[0];
      int feature_size = labels->dims()[1];
      int sz = batch_size * feature_size;

      auto* logits_data = logits->data<T>();
      auto* loss_data = loss->mutable_data<T>(place);
      auto* max_data = loss_data;
      auto* labels_data = labels->data<T>();
      auto* softmax_data = softmax->mutable_data<T>(place);
      auto* diff_exp_sum_data = diff_exp_sum.mutable_data<T>(
          framework::make_ddim({batch_size, 1}), place);
      auto stream = context.cuda_device_context().stream();

      /*
      int threads = 512;
      int grid1 = (batch_size + threads - 1)/threads;
      int grid2 = (sz + threads - 1)/threads;
      RowMaxReduce<<<grid1, threads, 0, stream>>>(logits_data, max_data,
      batch_size, feature_size);
      RowDiffExpReduce<<<grid1, threads, 0, stream>>>(logits_data, max_data,
      diff_exp_sum_data, batch_size, feature_size);
      SoftmaxImmediateKernel<<<grid2, threads, 0, stream>>>(logits_data,
      max_data, diff_exp_sum_data, softmax_data, batch_size, feature_size);
      CrossEntropyReduce<<<grid1, threads, 0, stream>>>(labels_data,
      softmax_data, loss_data, batch_size, feature_size);
      */

      auto logits_beg = thrust::device_pointer_cast(logits->data<T>());
      auto labels_beg = thrust::device_pointer_cast(labels->data<T>());
      auto loss_beg = thrust::device_pointer_cast(loss->mutable_data<T>(place));
      auto softmax_beg =
          thrust::device_pointer_cast(softmax->mutable_data<T>(place));
      auto max_beg = loss_beg;  // reuse loss space as max
      auto diff_exp_sum_beg =
          thrust::device_pointer_cast(diff_exp_sum.mutable_data<T>(
              framework::make_ddim({batch_size, 1}), place));

      auto row_beg = platform::MakeRowIndexMatrixIterator(0, feature_size);
      auto row_end = row_beg + sz;

      auto exec_policy = thrust::cuda::par.on(stream);
      thrust::reduce_by_key(exec_policy, row_beg, row_end, logits_beg,
                            thrust::make_discard_iterator(), max_beg,
                            thrust::equal_to<T>(),  // NOLINT
                            ValueClipMaxFunctor<T>());

      DiffExpFunctor<T> diff_exp_functor;
      auto repeat_max_beg = platform::MakeRepeatIterator(max_beg, feature_size);
      auto diff_exp_beg = platform::MakeZipTransformIterator(
          diff_exp_functor, logits_beg, repeat_max_beg);
      thrust::reduce_by_key(exec_policy, row_beg, row_end, diff_exp_beg,
                            thrust::make_discard_iterator(), diff_exp_sum_beg);

      auto repeat_diff_exp_sum_beg =
          platform::MakeRepeatIterator(diff_exp_sum_beg, feature_size);
      auto beg_tmp = platform::MakeZipIterator(logits_beg, repeat_max_beg,
                                               repeat_diff_exp_sum_beg);

      thrust::transform(exec_policy, beg_tmp,  // NOLINT
                        beg_tmp + sz, softmax_beg,
                        SoftmaxImmediateFunctor<T>());

      thrust::reduce_by_key(
          exec_policy, row_beg, row_end,
          platform::MakeZipTransformIterator(CrossEntropyImmediateFunctor<T>(),
                                             softmax_beg, labels_beg),
          thrust::make_discard_iterator(), loss_beg);
      /*
      auto beg = platform::MakeZipTransformIterator(functor, logits_beg,
      labels_beg, repeat_max_beg, repeat_diff_exp_sum_beg, softmax_beg);
      thrust::reduce_by_key(exec_policy, row_beg, row_end, beg,
      thrust::make_discard_iterator(), loss_beg);
      */
      /*
      auto eigen_prob = framework::EigenMatrix<T>::From(*logits);
      Tensor max_logits(tensor);

      auto eigen_labels = framework::EigenMatrix<T>::From(*labels);
      auto eigen_out = framework::EigenMatrix<T>::From(*out);
      auto mul_ret = -eigen_prob.log().cwiseMax(-kMaxTolerableValue)
                        .cwiseMin(kMaxTolerableValue) * eigen_labels;
      eigen_out.device(*ctx.eigen_device()) =
      mul_ret.sum(Eigen::array<size_t,1>({{1}}));
      */
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
    logit_grad->ShareDataWith(*context.Input<Tensor>("Softmax"));
    T* logit_grad_data = logit_grad->data<T>();

    const int batch_size = logit_grad->dims()[0];
    const int class_num = logit_grad->dims()[1];
    int block = 512;
    auto stream = context.cuda_device_context().stream();

    if (context.Attr<bool>("soft_label")) {
      int grid = (batch_size * class_num + block - 1) / block;
      const T* label_data = labels->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, batch_size, class_num);
    } else {
      int grid = (batch_size + block - 1) / block;
      const int64_t* label_data = labels->data<int64_t>();
      CrossEntropyGrad<T><<<grid, block, 0, stream>>>(
          logit_grad_data, label_data, batch_size, class_num);
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
REGISTER_OP_CUDA_KERNEL(softmax_with_cross_entropy,
                        ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
                        ops::SoftmaxWithCrossEntropyCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(softmax_with_cross_entropy_grad,
                        ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
                        ops::SoftmaxWithCrossEntropyGradCUDAKernel<double>);
