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

size_t GetBNMaskSpaceSize(uint32_t N, uint32_t C, uint32_t H, uint32_t W);

template <typename T>
void LaunchMaskedAddReluFwdKernel(const platform::CUDADeviceContext &dev_ctx,
                                  const T *x, const T *z, T *y, void *mask,
                                  size_t n);

template <typename T>
void LaunchMaskedReluBwdKernel(const platform::CUDADeviceContext &dev_ctx,
                               const T *dy, const void *mask, T *dx, size_t n);

bool TryLaunchFusedNCHWFP32BNTrainingKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, const float *z,
    const float *scale, const float *bias, float *y, float *save_mean,
    float *save_inv_variance, float *running_mean, float *running_variance,
    void *mask, uint32_t N, uint32_t C, uint32_t H, uint32_t W, double factor,
    double epsilon, bool need_relu);

}  // namespace operators
}  // namespace paddle
