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

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_CUDA_KERNEL(act_type, functor, grad_functor)    \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type,                                                             \
      ops::ActivationKernel<plat::CUDADeviceContext, ops::functor<float>>,  \
      ops::ActivationKernel<plat::CUDADeviceContext, ops::functor<double>>, \
      ops::ActivationKernel<plat::CUDADeviceContext,                        \
                            ops::functor<plat::float16>>);                  \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type##_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,   \
                                                 ops::grad_functor<float>>, \
      ops::ActivationGradKernel<plat::CUDADeviceContext,                    \
                                ops::grad_functor<double>>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_ACTIVATION_CUDA_KERNEL);
