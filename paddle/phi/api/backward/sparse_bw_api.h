// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {
namespace sparse {

// x_grad

PADDLE_API void abs_grad(const Tensor& x,
                         const Tensor& out_grad,
                         Tensor* x_grad);

// x_grad

PADDLE_API void acos_grad(const Tensor& x,
                          const Tensor& out_grad,
                          Tensor* x_grad);

// x_grad

PADDLE_API void acosh_grad(const Tensor& x,
                           const Tensor& out_grad,
                           Tensor* x_grad);

// x_grad, y_grad

PADDLE_API void add_grad(const Tensor& x,
                         const Tensor& y,
                         const Tensor& out_grad,
                         Tensor* x_grad,
                         Tensor* y_grad);

// input_grad, x_grad, y_grad

PADDLE_API void addmm_grad(const Tensor& input,
                           const Tensor& x,
                           const Tensor& y,
                           const Tensor& out_grad,
                           float alpha,
                           float beta,
                           Tensor* input_grad,
                           Tensor* x_grad,
                           Tensor* y_grad);

// x_grad

PADDLE_API void asin_grad(const Tensor& x,
                          const Tensor& out_grad,
                          Tensor* x_grad);

// x_grad

PADDLE_API void asinh_grad(const Tensor& x,
                           const Tensor& out_grad,
                           Tensor* x_grad);

// x_grad

PADDLE_API void atan_grad(const Tensor& x,
                          const Tensor& out_grad,
                          Tensor* x_grad);

// x_grad

PADDLE_API void atanh_grad(const Tensor& x,
                           const Tensor& out_grad,
                           Tensor* x_grad);

// x_grad, scale_grad, bias_grad

PADDLE_API void batch_norm_grad(const Tensor& x,
                                const Tensor& scale,
                                const Tensor& bias,
                                const paddle::optional<Tensor>& mean_out,
                                const paddle::optional<Tensor>& variance_out,
                                const Tensor& saved_mean,
                                const Tensor& saved_variance,
                                const paddle::optional<Tensor>& reserve_space,
                                const Tensor& out_grad,
                                float momentum,
                                float epsilon,
                                const std::string& data_format,
                                bool is_test,
                                bool use_global_stats,
                                bool trainable_statistics,
                                Tensor* x_grad,
                                Tensor* scale_grad,
                                Tensor* bias_grad);

// x_grad

PADDLE_API void cast_grad(const Tensor& x,
                          const Tensor& out_grad,
                          DataType value_dtype,
                          Tensor* x_grad);

// x_grad, kernel_grad

PADDLE_API void conv3d_grad(const Tensor& x,
                            const Tensor& kernel,
                            const Tensor& out,
                            const Tensor& rulebook,
                            const Tensor& counter,
                            const Tensor& out_grad,
                            const std::vector<int>& paddings,
                            const std::vector<int>& dilations,
                            const std::vector<int>& strides,
                            int groups,
                            bool subm,
                            const std::string& key,
                            Tensor* x_grad,
                            Tensor* kernel_grad);

// x_grad, y_grad

PADDLE_API void divide_grad(const Tensor& x,
                            const Tensor& y,
                            const Tensor& out,
                            const Tensor& out_grad,
                            Tensor* x_grad,
                            Tensor* y_grad);

// x_grad

// x_grad

PADDLE_API void expm1_grad(const Tensor& out,
                           const Tensor& out_grad,
                           Tensor* x_grad);

// x_grad

PADDLE_API void leaky_relu_grad(const Tensor& x,
                                const Tensor& out_grad,
                                float alpha,
                                Tensor* x_grad);

// x_grad

PADDLE_API void log1p_grad(const Tensor& x,
                           const Tensor& out_grad,
                           Tensor* x_grad);

// x_grad

PADDLE_API void mask_as_grad(const Tensor& x,
                             const Tensor& mask,
                             const Tensor& out_grad,
                             Tensor* x_grad);

// x_grad, y_grad

PADDLE_API void masked_matmul_grad(const Tensor& x,
                                   const Tensor& y,
                                   const Tensor& out_grad,
                                   Tensor* x_grad,
                                   Tensor* y_grad);

// x_grad, y_grad

PADDLE_API void matmul_grad(const Tensor& x,
                            const Tensor& y,
                            const Tensor& out_grad,
                            Tensor* x_grad,
                            Tensor* y_grad);

// x_grad

PADDLE_API void maxpool_grad(const Tensor& x,
                             const Tensor& rulebook,
                             const Tensor& counter,
                             const Tensor& out,
                             const Tensor& out_grad,
                             const std::vector<int>& kernel_sizes,
                             Tensor* x_grad);

// x_grad, y_grad

PADDLE_API void multiply_grad(const Tensor& x,
                              const Tensor& y,
                              const Tensor& out_grad,
                              Tensor* x_grad,
                              Tensor* y_grad);

// x_grad, vec_grad

PADDLE_API void mv_grad(const Tensor& x,
                        const Tensor& vec,
                        const Tensor& out_grad,
                        Tensor* x_grad,
                        Tensor* vec_grad);

// x_grad

PADDLE_API void pow_grad(const Tensor& x,
                         const Tensor& out_grad,
                         float factor,
                         Tensor* x_grad);

// x_grad

PADDLE_API void relu6_grad(const Tensor& out,
                           const Tensor& out_grad,
                           Tensor* x_grad);

// x_grad

PADDLE_API void relu_grad(const Tensor& out,
                          const Tensor& out_grad,
                          Tensor* x_grad);

// x_grad

PADDLE_API void reshape_grad(const Tensor& x,
                             const Tensor& out_grad,
                             Tensor* x_grad);

// x_grad

// x_grad

PADDLE_API void sin_grad(const Tensor& x,
                         const Tensor& out_grad,
                         Tensor* x_grad);

// x_grad

PADDLE_API void sinh_grad(const Tensor& x,
                          const Tensor& out_grad,
                          Tensor* x_grad);

// x_grad

PADDLE_API void softmax_grad(const Tensor& out,
                             const Tensor& out_grad,
                             int axis,
                             Tensor* x_grad);

// values_grad

PADDLE_API void sparse_coo_tensor_grad(const Tensor& indices,
                                       const Tensor& out_grad,
                                       Tensor* values_grad);

// x_grad

PADDLE_API void sqrt_grad(const Tensor& out,
                          const Tensor& out_grad,
                          Tensor* x_grad);

// x_grad

PADDLE_API void square_grad(const Tensor& x,
                            const Tensor& out_grad,
                            Tensor* x_grad);

// x_grad, y_grad

PADDLE_API void subtract_grad(const Tensor& x,
                              const Tensor& y,
                              const Tensor& out_grad,
                              Tensor* x_grad,
                              Tensor* y_grad);

// x_grad

PADDLE_API void sum_grad(const Tensor& x,
                         const Tensor& out_grad,
                         const IntArray& axis,
                         bool keepdim,
                         Tensor* x_grad);

// x_grad, scale_grad, bias_grad

PADDLE_API void sync_batch_norm_grad(
    const Tensor& x,
    const Tensor& scale,
    const Tensor& bias,
    const Tensor& saved_mean,
    const Tensor& saved_variance,
    const paddle::optional<Tensor>& reserve_space,
    const Tensor& out_grad,
    float momentum,
    float epsilon,
    const std::string& data_format,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    Tensor* x_grad,
    Tensor* scale_grad,
    Tensor* bias_grad);

// x_grad

PADDLE_API void tan_grad(const Tensor& x,
                         const Tensor& out_grad,
                         Tensor* x_grad);

// x_grad

PADDLE_API void tanh_grad(const Tensor& out,
                          const Tensor& out_grad,
                          Tensor* x_grad);

// x_grad

PADDLE_API void to_dense_grad(const Tensor& x,
                              const Tensor& out_grad,
                              Tensor* x_grad);

// x_grad

PADDLE_API void to_sparse_coo_grad(const Tensor& out_grad, Tensor* x_grad);

// x_grad

PADDLE_API void transpose_grad(const Tensor& out_grad,
                               const std::vector<int>& perm,
                               Tensor* x_grad);

// x_grad

PADDLE_API void values_grad(const Tensor& x,
                            const Tensor& out_grad,
                            Tensor* x_grad);

// query_grad, key_grad, value_grad

PADDLE_API void fused_attention_grad(const Tensor& query,
                                     const Tensor& key,
                                     const Tensor& value,
                                     const Tensor& softmax,
                                     const Tensor& out_grad,
                                     Tensor* query_grad,
                                     Tensor* key_grad,
                                     Tensor* value_grad);

// x_grad

PADDLE_API void slice_grad(const Tensor& x,
                           const Tensor& out_grad,
                           const IntArray& axes,
                           const IntArray& starts,
                           const IntArray& ends,
                           Tensor* x_grad);

}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
