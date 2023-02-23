/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/phi/backends/gpu/gpu_dnn.h"

namespace phi {

using GPUDNNDataLayout = phi::backends::gpu::DataLayout;
using PoolingMode = phi::backends::gpu::PoolingMode;
using ScopedPoolingDescriptor = phi::backends::gpu::ScopedPoolingDescriptor;
using ScopedTensorDescriptor = phi::backends::gpu::ScopedTensorDescriptor;

template <typename T>
using ScalingParamType =
    typename phi::backends::gpu::CudnnDataType<T>::ScalingParamType;

inline GPUDNNDataLayout GetLayoutFromStr(std::string data_format) {
  if (data_format == "NHWC") {
    return GPUDNNDataLayout::kNHWC;
  } else if (data_format == "NCHW") {
    return GPUDNNDataLayout::kNCHW;
  } else if (data_format == "NCDHW") {
    return GPUDNNDataLayout::kNCDHW;
  } else {
    return GPUDNNDataLayout::kNCDHW;
  }
}

}  // namespace phi
