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

#define EIGEN_USE_GPU
#include "paddle/fluid/operators/reduce_op.h"

namespace ops = paddle::operators;

#define REGISTER_REDUCE_GPU_KERNEL(reduce_type, functor, grad_functor)    \
  REGISTER_OP_CUDA_KERNEL(                                                \
      reduce_type, ops::ReduceKernel<paddle::platform::CUDADeviceContext, \
                                     float, ops::functor>,                \
      ops::ReduceKernel<paddle::platform::CUDADeviceContext, double,      \
                        ops::functor>,                                    \
      ops::ReduceKernel<paddle::platform::CUDADeviceContext, int,         \
                        ops::functor>,                                    \
      ops::ReduceKernel<paddle::platform::CUDADeviceContext, int64_t,     \
                        ops::functor>);                                   \
  REGISTER_OP_CUDA_KERNEL(                                                \
      reduce_type##_grad,                                                 \
      ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, float,   \
                            ops::grad_functor>,                           \
      ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, double,  \
                            ops::grad_functor>,                           \
      ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, int,     \
                            ops::grad_functor>,                           \
      ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, int64_t, \
                            ops::grad_functor>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_REDUCE_GPU_KERNEL);
