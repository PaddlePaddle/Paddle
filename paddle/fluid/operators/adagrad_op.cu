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
#include "paddle/fluid/operators/adagrad_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

namespace {

template <typename T, int block_size>
__global__ void SparseAdagradFunctorKernel(const T* grad, const int64_t* rows,
                                           const T* learning_rate, T* param,
                                           T* moment, int64_t row_numel,
                                           T epsilon) {
  const int ty = blockIdx.x;
  int tid = threadIdx.x;

  grad += ty * row_numel;
  param += rows[ty] * row_numel;
  moment += rows[ty] * row_numel;

  for (int index = tid; index < row_numel; index += block_size) {
    // Since index in rows of SelectedRows can be duplicate, we have to use
    // Atomic Operation to avoid concurrent write error.
    paddle::platform::CudaAtomicAdd(param + index,
                                    -1.0 * learning_rate[0] * grad[index] /
                                        (sqrt(moment[index]) + epsilon));
  }
}
}  // namespace

template <typename T>
struct SparseAdagradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::Tensor* moment, framework::Tensor* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_width = grad.value().dims()[1];
    math::scatter::MergeAdd<platform::CUDADeviceContext, T> merge_func;
    auto grad_merge = merge_func(context, grad);
    auto* grad_merge_data = grad_merge.mutable_value()->template data<T>();
    framework::Vector<int64_t> merge_rows(grad_merge.rows());
    // 2. m += g_m * g_m
    math::scatter::Mul<platform::CUDADeviceContext, T> sqare_func;
    auto grad_square = sqare_func(context, grad_merge, grad_merge);

    math::SelectedRowsAddToTensor<platform::CUDADeviceContext, T> functor;
    functor(context, grad_square, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();

    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid2(merge_rows.size(), 1);
    SparseAdagradFunctorKernel<
        T, block_size><<<grid2, threads, 0, context.stream()>>>(
        grad_merge_data, merge_rows.CUDAMutableData(context.GetPlace()), lr,
        param_data, moment_data, grad_width, epsilon);
  }
};

template struct SparseAdagradFunctor<platform::CUDADeviceContext, float>;
template struct SparseAdagradFunctor<platform::CUDADeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    adagrad, ops::AdagradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AdagradOpKernel<paddle::platform::CUDADeviceContext, double>);
