// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/adagrad_kernel.h"

#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/adagrad_kernel_impl.h"

namespace phi {

template <typename T, int block_size>
__global__ void MergeGradKernel(const T* grad,
                                const int64_t* grad_rows,
                                T* grad_merge,
                                const int64_t* grad_merge_rows,
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
__global__ void SparseAdagradFunctorKernel(const T* grad,
                                           const int64_t* rows,
                                           const T* learning_rate,
                                           T* param,
                                           T* moment,
                                           int64_t row_numel,
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

template <typename T>
struct SparseAdagradFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& context,
                  const phi::SelectedRows& grad,
                  const DenseTensor& learning_rate,
                  T epsilon,
                  DenseTensor* moment,
                  DenseTensor* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_width = grad.value().dims()[1];
    paddle::operators::math::scatter::MergeAdd<phi::GPUContext, T> merge_func;
    auto grad_merge = merge_func(context, grad);
    auto* grad_merge_data = grad_merge.mutable_value()->template data<T>();
    paddle::framework::Vector<int64_t> merge_rows(grad_merge.rows());
    // 2. m += g_m * g_m
    auto grad_square =
        SquareSelectedRows<phi::GPUContext, T>(context, grad_merge);

    paddle::operators::math::SelectedRowsAddToTensor<phi::GPUContext, T>
        functor;
    functor(context, grad_square, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();

    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid2(1, merge_rows.size());
    paddle::framework::MixVector<int64_t> mixv_merge_rows(&merge_rows);
    SparseAdagradFunctorKernel<
        T,
        256><<<grid2,
               threads,
               0,
               reinterpret_cast<const phi::GPUContext&>(context).stream()>>>(
        grad_merge_data,
        mixv_merge_rows.CUDAMutableData(context.GetPlace()),
        lr,
        param_data,
        moment_data,
        grad_width,
        epsilon);
    mixv_merge_rows.CopyToCPU();
  }
};

template struct SparseAdagradFunctor<phi::GPUContext, float>;
template struct SparseAdagradFunctor<phi::GPUContext, double>;

}  // namespace phi

PD_REGISTER_KERNEL(
    adagrad, GPU, ALL_LAYOUT, phi::AdagradDenseKernel, float, double) {}

PD_REGISTER_KERNEL(adagrad_dense_param_sparse_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdagradSparseKernel,
                   float,
                   double) {}
