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
#include "paddle/operators/activation_op.h"

namespace ops = paddle::operators;

REGISTER_OP_GPU_KERNEL(sigmoid,
                       ops::ActivationKernel<paddle::platform::GPUPlace, float,
                                             ops::SigmoidFunctor<float>>);
REGISTER_OP_GPU_KERNEL(
    sigmoid_grad, ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                                            ops::SigmoidGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(
    exp,
    ops::ActivationKernel<paddle::platform::GPUPlace, float, ops::ExpFunctor>);
REGISTER_OP_GPU_KERNEL(exp_grad,
                       ops::ActivationGradKernel<paddle::platform::GPUPlace,
                                                 float, ops::ExpGradFunctor>);
REGISTER_OP_GPU_KERNEL(relu,
                       ops::ActivationKernel<paddle::platform::GPUPlace, float,
                                             ops::ReluFunctor<float>>);
REGISTER_OP_GPU_KERNEL(
    relu_grad, ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                                         ops::ReluGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(
    tanh,
    ops::ActivationKernel<paddle::platform::GPUPlace, float, ops::TanhFunctor>);
REGISTER_OP_GPU_KERNEL(
    tanh_grad, ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                                         ops::TanhGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(
    sqrt,
    ops::ActivationKernel<paddle::platform::GPUPlace, float, ops::SqrtFunctor>);
REGISTER_OP_GPU_KERNEL(
    sqrt_grad, ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                                         ops::SqrtGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(
    abs,
    ops::ActivationKernel<paddle::platform::GPUPlace, float, ops::AbsFunctor>);
REGISTER_OP_GPU_KERNEL(abs_grad,
                       ops::ActivationGradKernel<paddle::platform::GPUPlace,
                                                 float, ops::AbsGradFunctor>);

REGISTER_OP_GPU_KERNEL(reciprocal,
                       ops::ActivationKernel<paddle::platform::GPUPlace, float,
                                             ops::ReciprocalFunctor<float>>);
REGISTER_OP_GPU_KERNEL(
    reciprocal_grad,
    ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                              ops::ReciprocalGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(
    log,
    ops::ActivationKernel<paddle::platform::GPUPlace, float, ops::LogFunctor>);
REGISTER_OP_GPU_KERNEL(
    log_grad, ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                                        ops::LogGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(square,
                       ops::ActivationKernel<paddle::platform::GPUPlace, float,
                                             ops::SquareFunctor>);
REGISTER_OP_GPU_KERNEL(
    square_grad, ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                                           ops::SquareGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(softsign,
                       ops::ActivationKernel<paddle::platform::GPUPlace, float,
                                             ops::SoftsignFunctor<float>>);
REGISTER_OP_GPU_KERNEL(
    softsign_grad, ops::ActivationGradKernel<paddle::platform::GPUPlace, float,
                                             ops::SoftsignGradFunctor<float>>);

REGISTER_OP_GPU_KERNEL(brelu,
                       ops::BReluKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(brelu_grad,
                       ops::BReluGradKernel<paddle::platform::GPUPlace, float>);

REGISTER_OP_GPU_KERNEL(soft_relu,
                       ops::SoftReluKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    soft_relu_grad, ops::SoftReluGradKernel<paddle::platform::GPUPlace, float>);

REGISTER_OP_GPU_KERNEL(pow, ops::PowKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(pow_grad,
                       ops::PowGradKernel<paddle::platform::GPUPlace, float>);

REGISTER_OP_GPU_KERNEL(stanh,
                       ops::STanhKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(stanh_grad,
                       ops::STanhGradKernel<paddle::platform::GPUPlace, float>);
