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

#include "paddle/fluid/operators/cross_entropy_op.h"
#include "paddle/phi/common/float16.h"

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(cross_entropy,
                          GPU,
                          ALL_LAYOUT,
                          ops::CrossEntropyOpKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
PD_REGISTER_STRUCT_KERNEL(cross_entropy_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::CrossEntropyGradientOpKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_STRUCT_KERNEL(cross_entropy2,
                          GPU,
                          ALL_LAYOUT,
                          ops::CrossEntropyOpKernel2,
                          float,
                          double,
                          phi::dtype::float16) {}
PD_REGISTER_STRUCT_KERNEL(cross_entropy_grad2,
                          GPU,
                          ALL_LAYOUT,
                          ops::CrossEntropyGradientOpKernel2,
                          float,
                          double,
                          phi::dtype::float16) {}
