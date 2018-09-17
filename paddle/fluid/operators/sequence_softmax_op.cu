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

#include "paddle/fluid/operators/sequence_softmax_op.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void sequence_softmax_kernel(const T *in_data, const size_t *ref_lod,
                                        const size_t src_hight, T *out_data) {
  for (size_t i = threadIdx.x; i < src_hight; i += blockDim.x) {
    size_t span = ref_lod[i + 1] - ref_lod[i];

    T result = 0;
    for (size_t j = 0; j < span; ++j) {
      T ele = in_data[ref_lod[i] + j];
      ele = exp(ele);
      result += ele;
      out_data[ref_lod[i] + j] = ele;
    }
    for (size_t j = 0; j < span; ++j) {
      T ele = out_data[ref_lod[i] + j];
      ele = ele / result;
      out_data[ref_lod[i] + j] = ele;
    }
  }
}

template <typename T>
__global__ void sequence_softmax_grad_kernel(const T *softmax_grad_data,
                                             const T *softmax_data,
                                             const size_t *ref_lod,
                                             const size_t src_hight,
                                             T *dx_data) {
  for (size_t i = threadIdx.x; i < src_hight; i += blockDim.x) {
    size_t span = ref_lod[i + 1] - ref_lod[i];
    T result = 0;
    for (size_t j = 0; j < span; ++j) {
      size_t idx = ref_lod[i] + j;
      T s_g_d = softmax_grad_data[idx];
      T s_d = softmax_data[idx];
      result += s_g_d * s_d;
    }

    for (size_t j = 0; j < span; ++j) {
      size_t idx = ref_lod[i] + j;
      T s_g_d = softmax_grad_data[idx];
      T s_d = softmax_data[idx];
      dx_data[idx] = (s_g_d - result) * s_d;
    }
  }
}

template <typename T>
struct SequenceSoftmaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext &context,
                  const LoDTensor &x,
                  const framework::Vector<size_t> &ref_lod, /*referenced lod*/
                  LoDTensor *out) {
    int hight = ref_lod.size() - 1;

    const int kThreadsPerBlock = 1024;
    int thread_x = kThreadsPerBlock;
    if (hight < kThreadsPerBlock) {  // block_cols is aligned by 32.
      thread_x = ((hight + 31) >> 5) << 5;
    }

    dim3 block_size(thread_x);
    dim3 grid_size(1);
    sequence_softmax_kernel<<<grid_size, block_size, 0, context.stream()>>>(
        x.data<T>(), ref_lod.CUDAData(context.GetPlace()), hight,
        out->mutable_data<T>(context.GetPlace()));
  }
};

template <typename T>
struct SequenceSoftmaxGradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext &context,
                  const LoDTensor &dout, const LoDTensor &out,
                  const framework::Vector<size_t> &ref_lod, /*referenced lod*/
                  LoDTensor *dx) {
    size_t hight = ref_lod.size() - 1;

    const int kThreadsPerBlock = 1024;
    int thread_x = kThreadsPerBlock;
    if (hight < kThreadsPerBlock) {  // block_cols is aligned by 32.
      thread_x = ((hight + 31) >> 5) << 5;
    }

    dim3 block_size(thread_x);
    dim3 grid_size(1);
    sequence_softmax_grad_kernel<<<grid_size, block_size, 0,
                                   context.stream()>>>(
        dout.data<T>(), out.data<T>(), ref_lod.CUDAData(context.GetPlace()),
        hight, dx->mutable_data<T>(context.GetPlace()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sequence_softmax,
    ops::SequenceSoftmaxKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceSoftmaxKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    sequence_softmax_grad,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CUDADeviceContext,
                                   double>);
