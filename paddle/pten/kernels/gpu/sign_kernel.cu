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

#include "paddle/pten/kernels/impl/sign_kernel_impl.h"
#include "paddle/pten/kernels/sign_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/float16.h"

using float16 = paddle::platform::float16;

PT_REGISTER_CTX_KERNEL(
    sign, GPU, ALL_LAYOUT, pten::Sign, float, double, float16) {}
