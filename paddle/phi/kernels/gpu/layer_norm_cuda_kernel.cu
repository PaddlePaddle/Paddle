// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/layer_norm_cuda_kernel.h"  // NOLINT
#include <cassert>
#include <vector>
#include "paddle/extension.h"
#include "paddle/common/exception.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"  // NOLINT

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
// #define CHECK_CUDA(x) PD_CHECK(!x.is_cpu(), #x " must be a CUDA tensor")

static void GetRowsCols(const std::vector<int64_t> &shape,
                        int *p_rows,
                        int *p_cols) {
  int rows = 1;
  for (int i = 0; i + 1 < shape.size(); ++i) {
    rows *= shape[i];
  }
  int cols = shape[shape.size() - 1];
  *p_rows = rows;
  *p_cols = cols;
}

template<typename Context>
std::vector<DenseTensor> RMSLnFwd(const Context& ctx,,
                                  const DenseTensor &x,
                                  const DenseTensor &scale,
                                  float epsilon) {
  const auto &scale_shape = scale.dims().size();
  const auto &x_shape = x.dims().size();
  PD_CHECK(scale_shape.size() == 1);
  PD_CHECK(scale_shape[0] == x_shape[x_shape.size() - 1]);

  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);

  auto place = x.place();
  auto y = phi::Empty(ctx, x_shape);
  auto invvar = phi::Empty(ctx, {rows});

  cuda_rms_norm<Context>(x, scale, rows, cols, epsilon, &y, &invvar);
  return {y, invvar};
}

template<typename Context>
std::vector<DenseTensor> LnFwd(const Context& ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &scale,
                                  const DenseTensor &bias,
                                  float epsilon) {
  const auto &scale_shape = scale.dims().size();
  const auto &bias_shape = bias.dims().size();
  const auto &x_shape = x.dims().size();
  PD_CHECK(scale_shape == bias_shape);
  PD_CHECK(scale_shape.size() == 1);
  PD_CHECK(scale_shape[0] == x_shape[x_shape.size() - 1]);

  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);

  auto place = x.place();
  auto y = phi::Empty(x_shape, scale.type(), place);
  auto mean = phi::Empty({rows}, paddle::DataType::FLOAT32, place);
  auto invvar = phi::EmptyLike(ctx.stream(), mean);

  cuda_layer_norm<Context>(x, scale, bias, rows, cols, epsilon, &y, &mean, &invvar);
  return {y, mean, invvar};
}

std::vector<std::vector<int64_t>> LnFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> bias_shape,
    float epsilon) {
  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);
  return {x_shape, {rows}, {rows}};
}

std::vector<std::vector<int64_t>> RMSLnFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    float epsilon) {
  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);
  return {x_shape, {rows}};
}

std::vector<paddle::DataType> LnFwdInferDtype(paddle::DataType x_dtype,
                                              paddle::DataType scale_dtype,
                                              paddle::DataType bias_dtype) {
  return {x_dtype, paddle::DataType::FLOAT32, paddle::DataType::FLOAT32};
}

std::vector<paddle::DataType> RMSLnFwdInferDtype(paddle::DataType x_dtype,
                                              paddle::DataType scale_dtype
                                              ) {
  return {x_dtype, paddle::DataType::FLOAT32};
}

template<typename Context>
std::vector<DenseTensor> LnBwd(const Context& ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &scale,
                                  const DenseTensor &bias,
                                  const DenseTensor &mean,
                                  const DenseTensor &invvar,
                                  const DenseTensor &dy,
                                  float epsilon) {
  int rows, cols;
  GetRowsCols(x.dims().size(), &rows, &cols);

  auto grad_x = phi::EmptyLike(ctx.stream(), x);
  auto grad_scale = phi::EmptyLike(ctx.stream(), scale);
  auto grad_bias = phi::EmptyLike(ctx.stream(), bias);

  cuda_layer_norm_gradient<Context>(x,
                           scale,
                           bias,
                           mean,
                           invvar,
                           dy,
                           rows,
                           cols,
                           epsilon,
                           &grad_x,
                           &grad_scale,
                           &grad_bias);
  return {grad_x, grad_scale, grad_bias};
}

template<typename Context>
std::vector<DenseTensor> RMSLnBwd(const Context& ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &scale,
                                  const DenseTensor &invvar,
                                  const DenseTensor &dy,
                                  float epsilon) {
  int rows, cols;
  GetRowsCols(x.dims().size(), &rows, &cols);

  auto grad_x = phi::EmptyLike(ctx.stream(), x);
  auto grad_scale = phi::EmptyLike(ctx.stream(), scale);

  cuda_rms_norm_gradient<Context>(x,
                           scale,
                           invvar,
                           dy,
                           rows,
                           cols,
                           epsilon,
                           &grad_x,
                           &grad_scale
                           );
  return {grad_x, grad_scale};
}

std::vector<std::vector<int64_t>> LnBwdInferShape(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> gamma_shape,
    std::vector<int64_t> beta_shape,
    std::vector<int64_t> mean_shape,
    std::vector<int64_t> invvar_shape,
    std::vector<int64_t> dout_shape,
    float epsilon) {
  return {input_shape, gamma_shape, beta_shape};
}

std::vector<std::vector<int64_t>> RMSLnBwdInferShape(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> gamma_shape,
    std::vector<int64_t> invvar_shape,
    std::vector<int64_t> dout_shape,
    float epsilon) {
  return {input_shape, gamma_shape};
}

} // namespace phi


// PD_BUILD_OP(fused_rms_norm)
//     .Inputs({"x", "scale"})
//     .Outputs({"y", "invvar"})
//     .Attrs({"epsilon: float"})
//     .SetKernelFn(PD_KERNEL(phi::RMSLnFwd))
//     .SetInferShapeFn(PD_INFER_SHAPE(phi::RMSLnFwdInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(phi::RMSLnFwdInferDtype));

PD_REGISTER_KERNEL(fused_rms_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::RMSLnFwd,
                   float) {}

// PD_REGISTER_KERNEL(fused_rms_norm,
//                    GPU,
//                    ALL_LAYOUT,
//                    phi::RMSLnFwd,
//                    float,
//                    double,
//                    int,
//                    int64_t,
//                    phi::dtype::float16,
//                    phi::dtype::complex<float>,
//                    phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(fused_rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RMSLnBwd,
                   float) {}
