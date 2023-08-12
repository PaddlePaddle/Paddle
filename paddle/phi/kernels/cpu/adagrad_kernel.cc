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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
#include "paddle/phi/kernels/impl/adagrad_kernel_impl.h"

namespace phi {

namespace {
size_t FindPos(const std::vector<int64_t>& rows, int64_t value) {
  return std::find(rows.begin(), rows.end(), value) - rows.begin();
}
}  // namespace

template <typename T>
struct DenseAdagradFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& param_t,
                  const DenseTensor& grad_t,
                  const DenseTensor& moment_t,
                  const DenseTensor& learning_rate,
                  const paddle::optional<DenseTensor>& master_param,
                  float epsilon_t,
                  bool multi_precision,
                  DenseTensor* param_out_tensor,
                  DenseTensor* moment_out_tensor,
                  DenseTensor* master_param_outs) {
    ctx.template Alloc<T>(param_out_tensor);
    ctx.template Alloc<T>(moment_out_tensor);

    T epsilon = static_cast<T>(epsilon_t);

    auto param = EigenVector<T>::Flatten(param_t);

    auto grad = EigenVector<T>::Flatten(grad_t);

    auto moment = EigenVector<T>::Flatten(moment_t);

    auto param_out = EigenVector<T>::Flatten(*param_out_tensor);
    auto moment_out = EigenVector<T>::Flatten(*moment_out_tensor);
    auto place = *ctx.eigen_device();

    moment_out.device(place) = moment + grad * grad;
    Eigen::DSizes<int, 1> m_dsize(static_cast<int>(moment_out_tensor->numel()));
    auto* lr = learning_rate.data<T>();
    param_out.device(place) =
        param - lr[0] * grad / (moment_out.sqrt() + epsilon);
  }
};

template <typename T>
struct SparseAdagradFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const phi::SelectedRows& grad,
                  const DenseTensor& learning_rate,
                  T epsilon,
                  DenseTensor* moment,
                  DenseTensor* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_width = grad.value().dims()[1];
    phi::funcs::scatter::MergeAdd<phi::CPUContext, T> merge_func;
    auto grad_merge = merge_func(context, grad);
    auto& merge_rows = grad_merge.rows();
    auto* grad_merge_data = grad_merge.mutable_value()->template data<T>();

    // 2. m += g_m * g_m
    auto grad_square =
        SquareSelectedRows<phi::CPUContext, T>(context, grad_merge);

    phi::funcs::SelectedRowsAddToTensor<phi::CPUContext, T> functor;
    functor(context, grad_square, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();

    for (size_t i = 0; i < merge_rows.size(); i++) {
      for (int64_t j = 0; j < grad_width; j++) {
        param_data[merge_rows[i] * grad_width + j] -=
            lr[0] * grad_merge_data[i * grad_width + j] /
            (std::sqrt(moment_data[merge_rows[i] * grad_width + j]) + epsilon);
      }
    }
  }
};

template struct SparseAdagradFunctor<phi::CPUContext, float>;
template struct SparseAdagradFunctor<phi::CPUContext, double>;
template struct DenseAdagradFunctor<phi::CPUContext, float>;
template struct DenseAdagradFunctor<phi::CPUContext, double>;

}  // namespace phi

PD_REGISTER_KERNEL(
    adagrad, CPU, ALL_LAYOUT, phi::AdagradDenseKernel, float, double) {}

PD_REGISTER_KERNEL(adagrad_dense_param_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AdagradSparseKernel,
                   float,
                   double) {}
