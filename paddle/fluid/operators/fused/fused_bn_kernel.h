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

#pragma once

#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

size_t GetFP32BNReserveSpaceSize(uint32_t N, uint32_t C, uint32_t H,
                                 uint32_t W);

void LaunchFP32MaskedAddReluFwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, const float *z,
    float *y, void *reserve_space, size_t n);

void LaunchFP32MaskedReluBwdKernel(const platform::CUDADeviceContext &dev_ctx,
                                   const float *dy, const void *reserve_space,
                                   float *dx, size_t n);

bool CanUseFusedNCHWFP32BNTrainingKernel(uint32_t N, uint32_t C, uint32_t H,
                                         uint32_t W);

void LaunchFusedNCHWFP32BNTrainingKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, const float *z,
    const float *scale, const float *bias, float *y, float *save_mean,
    float *save_inv_variance, float *running_mean, float *running_variance,
    void *reserve_space, uint32_t N, uint32_t C, uint32_t H, uint32_t W,
    double factor, double epsilon, bool need_relu);

}  // namespace operators
}  // namespace paddle
