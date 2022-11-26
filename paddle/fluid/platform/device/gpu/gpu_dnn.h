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

#pragma once

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/gpu/rocm/miopen_desc.h"
#include "paddle/phi/backends/gpu/rocm/miopen_helper.h"
#else  // CUDA
#include "paddle/phi/backends/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#endif

namespace paddle {
namespace platform {

using DataLayout = phi::backends::gpu::DataLayout;
using PoolingMode = phi::backends::gpu::PoolingMode;
template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
using ScopedTensorDescriptor = phi::backends::gpu::ScopedTensorDescriptor;
using ScopedRNNTensorDescriptor = phi::backends::gpu::ScopedRNNTensorDescriptor;
using ScopedDropoutDescriptor = phi::backends::gpu::ScopedDropoutDescriptor;
using ScopedRNNDescriptor = phi::backends::gpu::ScopedRNNDescriptor;
using ScopedFilterDescriptor = phi::backends::gpu::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor =
    phi::backends::gpu::ScopedConvolutionDescriptor;
using ScopedPoolingDescriptor = phi::backends::gpu::ScopedPoolingDescriptor;
using ScopedSpatialTransformerDescriptor =
    phi::backends::gpu::ScopedSpatialTransformerDescriptor;
using ScopedActivationDescriptor =
    phi::backends::gpu::ScopedActivationDescriptor;

}  // namespace platform
}  // namespace paddle

#endif
