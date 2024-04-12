/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_elemwise_activation_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(fused_elemwise_activation,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedElemwiseActivationKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
PD_REGISTER_STRUCT_KERNEL(fused_elemwise_activation_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedElemwiseActivationGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_STRUCT_KERNEL(fused_elemwise_add_activation,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedElemwiseAddActivationKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
PD_REGISTER_STRUCT_KERNEL(fused_elemwise_add_activation_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedElemwiseAddActivationGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
