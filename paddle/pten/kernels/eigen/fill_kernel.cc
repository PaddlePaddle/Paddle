/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/kernels/fill_kernel.h"
#include "paddle/pten/kernels/eigen/fill.h"

#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/backends/all_context.h"

PT_REGISTER_CTX_KERNEL(full,
                       CPU,
                       ALL_LAYOUT,
                       pten::Full,
                       float,
                       double,
                       uint8_t,
                       int16_t,
                       int,
                       int64_t,
                       bool,
                       paddle::platform::float16,
                       paddle::platform::bfloat16,
                       paddle::platform::complex<float>,
                       paddle::platform::complex<double>) {}

PT_REGISTER_CTX_KERNEL(full_like,
                       CPU,
                       ALL_LAYOUT,
                       pten::FullLike,
                       float,
                       double,
                       int,
                       int64_t,
                       bool,
                       paddle::platform::float16) {}
