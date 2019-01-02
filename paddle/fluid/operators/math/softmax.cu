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
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/reductions.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
void SoftmaxCUDNNFunctor<T>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor* X,
    framework::Tensor* Y) {
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor xDesc;
  ScopedTensorDescriptor yDesc;
  std::vector<int> cudnn_tensor_dims = framework::vectorize2int(X->dims());
  DataLayout layout = DataLayout::kNCHW;
  if (cudnn_tensor_dims.size() == 5) {
    layout = DataLayout::kNCDHW;
  }
  // NOTE(*) : cudnn softmax only support >= 4D Tensor,
  // fill 1 at unused dims
  if (cudnn_tensor_dims.size() <= 2) {
    cudnn_tensor_dims.resize(4, 1);
  }
  cudnnTensorDescriptor_t cudnn_x_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_y_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  CUDNN_ENFORCE(platform::dynload::cudnnSoftmaxForward(
      context.cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_INSTANCE, CudnnDataType<T>::kOne(), cudnn_x_desc,
      X->data<T>(), CudnnDataType<T>::kZero(), cudnn_y_desc,
      Y->mutable_data<T>(context.GetPlace())));
}

template <typename T>
void SoftmaxGradCUDNNFunctor<T>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor* Y,
    const framework::Tensor* YGrad, framework::Tensor* XGrad) {
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor yDesc;
  ScopedTensorDescriptor dyDesc;
  ScopedTensorDescriptor dxDesc;
  std::vector<int> cudnn_tensor_dims = framework::vectorize2int(Y->dims());
  DataLayout layout = DataLayout::kNCHW;
  if (cudnn_tensor_dims.size() == 5) {
    layout = DataLayout::kNCDHW;
  }
  // NOTE(*) : cudnn softmax only support >= 4D Tensor,
  // fill 1 at unused dims
  if (cudnn_tensor_dims.size() <= 2) {
    cudnn_tensor_dims.resize(4, 1);
  }
  cudnnTensorDescriptor_t cudnn_y_desc =
      yDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_xgrad_desc =
      dxDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_ygrad_desc =
      dyDesc.descriptor<T>(layout, cudnn_tensor_dims);
  CUDNN_ENFORCE(platform::dynload::cudnnSoftmaxBackward(
      context.cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_INSTANCE, CudnnDataType<T>::kOne(), cudnn_y_desc,
      Y->data<T>(), cudnn_ygrad_desc, YGrad->data<T>(),
      CudnnDataType<T>::kZero(), cudnn_xgrad_desc,
      XGrad->mutable_data<T>(context.GetPlace())));
}

template class SoftmaxCUDNNFunctor<platform::float16>;
template class SoftmaxCUDNNFunctor<float>;
template class SoftmaxCUDNNFunctor<double>;
template class SoftmaxGradCUDNNFunctor<float>;
template class SoftmaxGradCUDNNFunctor<double>;
template class SoftmaxGradCUDNNFunctor<platform::float16>;

template class SoftmaxFunctor<platform::CUDADeviceContext, platform::float16,
                              false>;
template class SoftmaxFunctor<platform::CUDADeviceContext, platform::float16,
                              true>;
template class SoftmaxFunctor<platform::CUDADeviceContext, float, false>;
template class SoftmaxFunctor<platform::CUDADeviceContext, double, false>;
template class SoftmaxFunctor<platform::CUDADeviceContext, float, true>;
template class SoftmaxFunctor<platform::CUDADeviceContext, double, true>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, double>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext,
                                  platform::float16>;

// NOTE: upgrade value type for float16 and float32
static __device__ __forceinline__ float real_exp(platform::float16 x) {
  return ::Eigen::numext::exp(static_cast<float>(x));
}
static __device__ __forceinline__ double real_exp(float x) {
  return exp(static_cast<double>(x));
}
static __device__ __forceinline__ double real_exp(double x) { return exp(x); }

static __device__ __forceinline__ platform::float16 real_log(
    platform::float16 x) {
  return ::Eigen::numext::log(x);
}
static __device__ __forceinline__ float real_log(float x) { return logf(x); }
static __device__ __forceinline__ double real_log(double x) { return log(x); }

template <typename T, typename ACCURATE_T>
struct SubtractAndExpFunctor {
  __host__ __device__ SubtractAndExpFunctor(const T* logits,
                                            const T* max_logits,
                                            const int num_cols)
      : logits_(logits), max_logits_(max_logits), num_cols_(num_cols) {}

  __host__ __device__ ACCURATE_T operator()(const int gid) const {
    return real_exp(logits_[gid] - (max_logits_[gid / num_cols_]));
  }

  const T* logits_;
  const T* max_logits_;
  const int num_cols_;
};

template <typename T, typename ACCURATE_T>
__global__ void GenerateProb(const T* logits, const ACCURATE_T* sum_probs,
                             const T* max_logits, ACCURATE_T* output,
                             const int num_rows, const int num_cols,
                             const bool log_space) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int row = tid / num_cols;
  const int col = tid % num_cols;

  if (row < num_rows && col < num_cols) {
    if (log_space) {
      output[tid] = static_cast<ACCURATE_T>(logits[tid] - (max_logits[row])) -
                    real_log(sum_probs[row]);
    } else {
      output[tid] =
          real_exp(logits[tid] - (max_logits[row])) / (sum_probs[row]);
    }
  }
}

template <typename T, typename ACCURATE_T>
void SoftmaxCudaAccurateFunctor<T, ACCURATE_T>::operator()(
    const platform::CUDADeviceContext& ctx, const framework::Tensor* X,
    framework::Tensor* Y, bool log_space) {
  Tensor max_logits, sum_probs;
  framework::DDim max_dims({X->dims()[0], 1});
  max_logits.Resize(max_dims);
  sum_probs.Resize(X->dims());
  auto* max_logits_data = max_logits.mutable_data<T>(ctx.GetPlace());
  auto* logits_data = X->data<T>();
  auto* sum_probs_data = sum_probs.mutable_data<ACCURATE_T>(ctx.GetPlace());
  int num_rows = X->dims()[0];
  int num_cols = X->dims()[1];

  // Use ACCURATE_T for internal exp space
  auto tmp_allocation_ptr =
      platform::DeviceTemporaryAllocator::Instance().Get(ctx).Allocate(
          Y->numel() * sizeof(ACCURATE_T));
  ACCURATE_T* acc_out_data =
      static_cast<ACCURATE_T*>(tmp_allocation_ptr->ptr());

  // RowReduce to find max along every batch (axis 1)
  cub::Max max_op;
  paddle::operators::math::LaunchRowReduction<T, T*, const T*, cub::Max>(
      &ctx, max_logits_data, logits_data, num_rows, num_cols, max_op,
      std::numeric_limits<T>::lowest());  // NOLINT

  // RowReduce to calculate exp(x - x.max())
  cub::CountingInputIterator<int> counting_iterator(0);
  typedef cub::TransformInputIterator<ACCURATE_T,
                                      SubtractAndExpFunctor<T, ACCURATE_T>,
                                      cub::CountingInputIterator<int>>
      InputIterType;

  InputIterType input_itr(counting_iterator,
                          SubtractAndExpFunctor<T, ACCURATE_T>(
                              logits_data, max_logits_data, num_cols));

  cub::Sum sum_op;
  paddle::operators::math::LaunchRowReduction<ACCURATE_T, ACCURATE_T*,
                                              InputIterType, cub::Sum>(
      &ctx, sum_probs_data, input_itr, num_rows, num_cols, sum_op,
      static_cast<ACCURATE_T>(0.0f));

  const int num_threads = 128;
  const int num_blocks =
      (num_rows * num_cols + num_threads - 1) / num_threads;  // divide ceil

  GenerateProb<T, ACCURATE_T><<<num_blocks, num_threads, 0, ctx.stream()>>>(
      logits_data, sum_probs_data, max_logits_data, acc_out_data, num_rows,
      num_cols, log_space);

  // transform data type, if log_space = false, may lost accuracy
  T* out_data = Y->mutable_data<T>(ctx.GetPlace());
  platform::Transform<platform::CUDADeviceContext> trans;
  trans(ctx, acc_out_data, acc_out_data + Y->numel(), out_data,
        CastOpTransformFunctor<ACCURATE_T, T>());
}

template class SoftmaxCudaAccurateFunctor<platform::float16, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
