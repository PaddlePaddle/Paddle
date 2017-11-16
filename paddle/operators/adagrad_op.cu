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

#define EIGEN_USE_GPU
#include "paddle/operators/adagrad_op.h"
#include "paddle/operators/math/selected_rows_functor.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {

namespace {

template <typename T, int block_size>
__global__ void MergeGradKernel(const T* grad, const int64_t* grad_rows,
                                T* grad_merge, const int64_t* grad_merge_rows,
                                size_t grad_merge_rows_size,
                                int64_t row_numel) {
  const int ty = blockIdx.y;
  int tid = threadIdx.x;
  __shared__ size_t grad_merge_idx;

  if (tid == 0) {
    for (size_t i = 0; i < grad_merge_rows_size; i++) {
      if (grad_rows[ty] == grad_merge_rows[i]) {
        grad_merge_idx = i;
      }
    }
  }

  __syncthreads();

  grad += ty * row_numel;
  grad_merge += grad_merge_idx * row_numel;
  for (int index = tid; index < row_numel; index += block_size) {
    paddle::platform::CudaAtomicAdd(grad_merge + index, grad[index]);
  }
}

template <typename T, int block_size>
__global__ void SparseAdagradFunctorKernel(const T* grad, const int64_t* rows,
                                           const T* learning_rate, T* param,
                                           T* moment, int64_t row_numel,
                                           T epsilon) {
  const int ty = blockIdx.y;
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
struct SparseAdagradFunctor<platform::GPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::Tensor* moment, framework::Tensor* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_rows = grad.rows();
    std::set<int64_t> row_set(grad_rows.begin(), grad_rows.end());
    std::vector<int64_t> merge_rows(row_set.begin(), row_set.end());

    auto grad_width = grad.value().dims()[1];
    std::unique_ptr<framework::SelectedRows> grad_merge{
        new framework::SelectedRows()};
    grad_merge->set_rows(merge_rows);
    grad_merge->set_height(grad.height());
    grad_merge->mutable_value()->mutable_data<T>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), grad_width}),
        context.GetPlace());

    math::SetConstant<platform::GPUPlace, T> constant_functor;
    constant_functor(context, grad_merge->mutable_value(), 0.0);

    auto* grad_merge_data = grad_merge->mutable_value()->data<T>();
    auto* grad_data = grad.value().data<T>();

    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid1(1, grad_rows.size());

    MergeGradKernel<
        T, 256><<<grid1, threads, 0,
                  reinterpret_cast<const platform::CUDADeviceContext&>(context)
                      .stream()>>>(grad_data, grad.rows().data(),
                                   grad_merge_data, grad_merge->rows().data(),
                                   grad_merge->rows().size(), grad_width);

    // 2. m += g_m * g_m
    std::unique_ptr<framework::SelectedRows> grad_square{
        new framework::SelectedRows()};
    grad_square->set_rows(grad_merge->rows());
    grad_square->set_height(grad_merge->height());
    grad_square->mutable_value()->mutable_data<T>(grad_merge->value().dims(),
                                                  context.GetPlace());
    auto gs =
        framework::EigenVector<T>::Flatten(*(grad_square->mutable_value()));
    auto gm = framework::EigenVector<T>::Flatten(grad_merge->value());
    gs.device(*context.GetEigenDevice<platform::GPUPlace>()) = gm * gm;

    math::SelectedRowsAddToTensor<platform::GPUPlace, T> functor;
    functor(context, *grad_square, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();

    dim3 grid2(1, merge_rows.size());
    SparseAdagradFunctorKernel<
        T, 256><<<grid2, threads, 0,
                  reinterpret_cast<const platform::CUDADeviceContext&>(context)
                      .stream()>>>(grad_merge_data, grad_merge->rows().data(),
                                   lr, param_data,
                                   moment_data, grad_width, epsilon);
  }
};

template struct SparseAdagradFunctor<platform::GPUPlace, float>;
template struct SparseAdagradFunctor<platform::GPUPlace, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(
    adagrad, ops::AdagradOpKernel<paddle::platform::GPUPlace, float>,
    ops::AdagradOpKernel<paddle::platform::GPUPlace, double>);
