/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/activation_op.cu.h"

namespace paddle {
namespace operators {

#define DEFINE_ACTIVATION_CUDA_KERNEL(op_name, functor, grad_functor) \
  template <typename T, typename DeviceContext>                       \
  class op_name##CudaKernel                                           \
      : public ActivationCudaKernel<DeviceContext, functor<T>> {};    \
                                                                      \
  template <typename T, typename DeviceContext>                       \
  class op_name##GradCudaKernel                                       \
      : public ActivationGradCudaKernel<DeviceContext, grad_functor<T>> {};

DEFINE_ACTIVATION_CUDA_KERNEL(SoftRelu,
                              CudaSoftReluFunctor,
                              CudaSoftReluGradFunctor)

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(soft_relu,
                          GPU,
                          ALL_LAYOUT,
                          ops::SoftReluCudaKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
PD_REGISTER_STRUCT_KERNEL(soft_relu_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::SoftReluGradCudaKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
