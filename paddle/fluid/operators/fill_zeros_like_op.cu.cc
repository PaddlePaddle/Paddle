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

#include "paddle/fluid/operators/fill_zeros_like_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(fill_zeros_like,
                          GPU,
                          ALL_LAYOUT,
                          ops::FillZerosLikeKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          plat::float16,
                          bool,
                          plat::complex<float>,
                          plat::complex<double>) {}

PD_REGISTER_STRUCT_KERNEL(fill_zeros_like2,
                          GPU,
                          ALL_LAYOUT,
                          ops::FillZerosLikeKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          plat::float16,
                          bool,
                          plat::complex<float>,
                          plat::complex<double>) {}
