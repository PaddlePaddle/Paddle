// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/split_kernel.h"

#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/split_kernel_impl.h"

PD_REGISTER_KERNEL(split,
                   GPU,
                   ALL_LAYOUT,
                   phi::SplitKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(split_with_num,
                   GPU,
                   ALL_LAYOUT,
                   phi::SplitWithNumKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
