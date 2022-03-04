// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/dist_kernel.h"
#include "paddle/phi/kernels/impl/dist_kernel_impl.h"

#ifdef PADDLE_WITH_HIP
// Eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:922
// do not support double in HIPCC platform (Eigen3 to be fixed)
PD_REGISTER_KERNEL(dist, GPU, ALL_LAYOUT, phi::DistKernel, float) {}
#else
PD_REGISTER_KERNEL(dist, GPU, ALL_LAYOUT, phi::DistKernel, float, double) {}
#endif
