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

#include <algorithm>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/math/quant.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, bool DoRelu>
__global__ void InplaceAddReluKernel(const T* bias, T* data, int M, int N) {
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    int index = i * N + threadIdx.x;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      T tmp = data[index] + bias[j];
      if (DoRelu) {
        data[index] = (tmp > 0) ? tmp : 0;
      } else {
        data[index] = tmp;
      }
      index += blockDim.x;
    }
  }
}

template <typename T>
class FCFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context, const int M,
                  const int N, const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false,
                  bool padding_weights = false) {
    PADDLE_ENFORCE_EQ(
        padding_weights, false,
        platform::errors::PermissionDenied(
            "Weight padding in fc can not be used in GPU scope."));
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
    blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), X, K, W, N,
              static_cast<T>(0.0), Y, N);
    if (B == NULL) {
      return;
    }

    const int kThreadsPerBlock = 1024;
    int max_threads = context.GetMaxPhysicalThreadCount();
    int num_threads = std::min(kThreadsPerBlock, (((N + 31) >> 5) << 5));
    int num_blocks = std::max(max_threads / num_threads, 1);
    if (relu) {
      InplaceAddReluKernel<
          T, true><<<num_blocks, num_threads, 0, context.stream()>>>(B, Y, M,
                                                                     N);
    } else {
      InplaceAddReluKernel<
          T, false><<<num_blocks, num_threads, 0, context.stream()>>>(B, Y, M,
                                                                      N);
    }
  }
};

void FCInt8Functor<platform::CUDADeviceContext>::operator()(
    const platform::CUDADeviceContext& context, int M, int N, int K,
    const framework::Tensor& in, const framework::Tensor& W,
    framework::Tensor* Y, float in_scale, std::vector<float> weight_scale,
    const framework::Tensor& B, bool relu, bool weight_pass) {
  PADDLE_ENFORCE_EQ(weight_pass, false,
                    platform::errors::PermissionDenied(
                        "Weight padding in fc can not be used in GPU scope."));

  framework::Tensor x_int8;
  // modify
  x_int8.Resize(in.dims());
  x_int8.mutable_data<int8_t>(context.GetPlace());

  QuantFp32ToInt8Functor<platform::CUDADeviceContext> quant_func;
  quant_func(context, in, in_scale / 127., &x_int8);
  // the float here represents the output type
  const int8_t* x_int8_data = x_int8.data<int8_t>();
  const int8_t* w_int8_data = W.data<int8_t>();

  if (N % 4 == 0) {
    framework::Tensor x_int8, out_int8;
    out_int8.Resize(Y->dims());
    int32_t* out_int8_data = out_int8.mutable_data<int32_t>(context.GetPlace());
    int32_t alpha = 1;
    int32_t beta = 0;
    float scale = static_cast<float>(in_scale * weight_scale[0] / 127. / 127.);
    GEMMINT8Functor<platform::CUDADeviceContext> gemm_int8_func;
    gemm_int8_func(context, false, false, M, N, K, alpha, x_int8_data, K,
                   w_int8_data, N, beta, out_int8_data, N);
    INT32ToFP32Functor<platform::CUDADeviceContext> int32_to_fp32_func;
    int32_to_fp32_func(context, out_int8, Y, scale);
  } else {
    float* y_data = Y->mutable_data<float>(context.GetPlace());
    float alpha = static_cast<float>(in_scale * weight_scale[0] / 127. / 127.);
    float beta = 0.0f;
    GEMMINT8Functor<platform::CUDADeviceContext> gemm_int8_func;
    gemm_int8_func(context, false, false, M, N, K, alpha, x_int8_data, K,
                   w_int8_data, N, beta, y_data, N);
  }

  const float* bias = B.data<float>();
  if (bias == NULL) {
    return;
  }
  float* y_data = Y->mutable_data<float>(context.GetPlace());
  const int kThreadsPerBlock = 1024;
  int max_threads = context.GetMaxPhysicalThreadCount();
  int num_threads = std::min(kThreadsPerBlock, (((N + 31) >> 5) << 5));
  int num_blocks = std::max(max_threads / num_threads, 1);
  if (relu) {
    InplaceAddReluKernel<
        float, true><<<num_blocks, num_threads, 0, context.stream()>>>(
        bias, y_data, M, N);
  } else {
    InplaceAddReluKernel<
        float, false><<<num_blocks, num_threads, 0, context.stream()>>>(
        bias, y_data, M, N);
  }
}

template class FCFunctor<platform::CUDADeviceContext, float>;
template class FCFunctor<platform::CUDADeviceContext, double>;
template class QuantFp32ToInt8Functor<platform::CUDADeviceContext>;
template class GEMMINT8Functor<platform::CUDADeviceContext>;
template class INT32ToFP32Functor<platform::CUDADeviceContext>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
