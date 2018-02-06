/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/sequence_pooling.h"

namespace paddle {
namespace operators {
namespace math {

#define FLT_MAX __FLT_MAX__

template <typename T>
__global__ void KeMaxSequencePool(const T* input, const size_t* starts,
                                  T* output, int* index, int64_t num_seq,
                                  int64_t dim) {
  int dim_idx = threadIdx.x;
  int seq_id = blockIdx.x;
  if (seq_id >= num_seq) return;
  size_t start = starts[seq_id];
  size_t end = starts[seq_id + 1];

  for (int64_t i = dim_idx; i < dim; i += blockDim.x) {
    T max_val = static_cast<T>(-FLT_MAX);
    int max_id = -1;
    for (size_t step_id = start; step_id < end; step_id++) {
      if (max_val < input[step_id * dim + i]) {
        max_val = input[step_id * dim + i];
        max_id = step_id;
      }
    }
    output[seq_id * dim + i] = max_val;
    index[seq_id * dim + i] = max_id;
  }
}

template <typename T>
class MaxSeqPoolFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& input, framework::Tensor* output,
                  framework::Tensor* index) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto idx_dims = index->dims();
    PADDLE_ENFORCE_GT(in_dims.size(), static_cast<int64_t>(1));
    PADDLE_ENFORCE_GT(out_dims.size(), 1);
    for (int64_t i = 1; i < in_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_dims[i], out_dims[i]);
    }
    PADDLE_ENFORCE_EQ(idx_dims, out_dims);

    auto starts = input.lod()[0];
    const T* in_data = input.data<T>();
    T* out_data = output->data<T>();
    int* max_index = index->data<int>();

    int64_t num_seq = out_dims[0];
    int64_t dim = output->numel() / num_seq;

    dim3 threads(256, 1);
    dim3 grid(num_seq, 1);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(context).stream();
    KeMaxSequencePool<T><<<grid, threads, 0, stream>>>(
        in_data, starts.data(), out_data, max_index, num_seq, dim);
  }
};

template <typename T>
__global__ void KeMaxSequencePoolGrad(const T* out_grad, const int* max_index,
                                      T* in_grad, int64_t num_seq,
                                      int64_t dim) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int col_idx = idx % dim;
  if (idx < num_seq * dim) {
    int step_id = max_index[idx];
    in_grad[step_id * dim + col_idx] = out_grad[idx];
  }
}

template <typename T>
class MaxSeqPoolGradFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& out_grad,
                  const framework::Tensor& index,
                  framework::LoDTensor* in_grad) {
    auto og_dims = out_grad.dims();
    auto idx_dims = index.dims();
    auto ig_dims = in_grad->dims();
    PADDLE_ENFORCE_GT(og_dims.size(), static_cast<int64_t>(1));
    PADDLE_ENFORCE_GT(ig_dims.size(), static_cast<int64_t>(1));
    for (int64_t i = 1; i < og_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(og_dims[i], ig_dims[i]);
    }
    PADDLE_ENFORCE_EQ(idx_dims, og_dims);

    const T* og_data = out_grad.data<T>();
    const int* max_index = index.data<int>();
    T* ig_data = in_grad->data<T>();

    SetConstant<platform::GPUPlace, T> set_zero;
    set_zero(context, in_grad, static_cast<T>(0.0));
    int64_t num_seq = og_dims[0];
    int64_t dim = out_grad.numel() / num_seq;

    unsigned int blocks = (num_seq * dim + 128 - 1) / 128;
    dim3 threads(128, 1);
    dim3 grid(blocks, 1);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(context).stream();
    KeMaxSequencePoolGrad<T><<<grid, threads, 0, stream>>>(
        og_data, max_index, ig_data, num_seq, dim);
  }
};

template class MaxSeqPoolFunctor<platform::GPUPlace, float>;
template class MaxSeqPoolFunctor<platform::GPUPlace, double>;
template class MaxSeqPoolGradFunctor<platform::GPUPlace, float>;
template class MaxSeqPoolGradFunctor<platform::GPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
