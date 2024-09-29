// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/l1_norm_kernel.h"
namespace phi {
// Out = sum(abs(X))
template <typename T, typename Context>
void L1NormKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto x_tmp = phi::EigenVector<T>::Flatten(x);
  auto out_tmp = phi::EigenScalar<T>::From(*out);
  auto& dev = *dev_ctx.eigen_device();
  phi::funcs::EigenL1Norm<std::decay_t<decltype(dev)>, T>::Eval(
      dev, out_tmp, x_tmp);
}
// dX = dout * sign(X)
template <typename T, typename Context>
void L1NormGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(out_grad.numel(),
                    1,
                    common::errors::InvalidArgument(
                        "Input(GRAD@Out) of L1NormGradOp should be a scalar."));
  dev_ctx.template Alloc<T>(x_grad);
  auto x_eigen = phi::EigenVector<T>::Flatten(x);
  auto d_out_eigen = phi::EigenVector<T>::Flatten(out_grad);
  auto dx_eigen = phi::EigenVector<T>::Flatten(*x_grad);
  auto& dev = *dev_ctx.eigen_device();
  Eigen::DSizes<Eigen::DenseIndex, 1> x_dsize(x.numel());
  phi::funcs::EigenL1NormGrad<std::decay_t<decltype(dev)>, T>::Eval(
      dev, dx_eigen, d_out_eigen, x_eigen, x_dsize);
}
}  // namespace phi
PD_REGISTER_KERNEL(
    l1_norm_grad, GPU, ALL_LAYOUT, phi::L1NormGradKernel, float) {}
