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

#pragma once

#include "paddle/pten/core/kernel_registry.h"

// TODO(chenweihang) After the kernel is split into a single file,
// the kernel declare statement is automatically generated according to the
// file name of the kernel, and this header file will be removed

PT_DECLARE_KERNEL(full_like, CPU, ALL_LAYOUT);
PT_DECLARE_KERNEL(dot, CPU, ALL_LAYOUT);
PT_DECLARE_KERNEL(flatten, CPU, ALL_LAYOUT);
PT_DECLARE_KERNEL(sign, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_KERNEL(full_like, CUDA, ALL_LAYOUT);
PT_DECLARE_KERNEL(dot, CUDA, ALL_LAYOUT);
PT_DECLARE_KERNEL(flatten, CUDA, ALL_LAYOUT);
PT_DECLARE_KERNEL(sign, CUDA, ALL_LAYOUT);
#endif

#ifdef PADDLE_WITH_XPU
PT_DECLARE_KERNEL(flatten, XPU, ALL_LAYOUT);
#endif
