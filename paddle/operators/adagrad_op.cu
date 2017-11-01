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

namespace paddle {
namespace operators {

namespace {
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
    std::unique_ptr<framework::SelectedRows> grad_square{
        new framework::SelectedRows(grad.rows(), grad.height())};
    grad_square->mutable_value()->mutable_data<T>(grad.value().dims(),
                                                  context.GetPlace());
    auto gs =
        framework::EigenVector<T>::Flatten(*(grad_square->mutable_value()));
    auto g = framework::EigenVector<T>::Flatten(grad.value());
    gs.device(*context.GetEigenDevice<platform::GPUPlace>()) = g * g;

    math::SelectedRowsAddToTensor<platform::GPUPlace, T> functor;
    functor(context, *grad_square, moment);

    auto grad_rows = grad.rows();
    auto grad_rows_size = grad_rows.size();

    int64_t grad_row_numel = grad.value().numel() / grad_rows_size;

    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();
    auto* grad_data = grad.value().data<T>();

    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid(1, in_rows.size());
    SparseAdagradFunctorKernel<
        T, 256><<<grid, threads, 0,
                  reinterpret_cast<const platform::CUDADeviceContext&>(context)
                      .stream()>>>(grad_data, grad.rows().data(),
                                   learning_rate.data<T>(), param_data,
                                   moment_data, grad_row_numel, epsilon);
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
