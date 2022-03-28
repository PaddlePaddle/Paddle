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

#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/infrt/dialect/infrt/common/types.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/tensor/phi/tensor_map.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace kernel {
namespace phi {

::phi::DenseTensor CreateDenseTensor(
    const ::phi::CPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<::infrt::PrecisionType> precision);

::phi::DenseTensor CreateInitedDenseTensorF32(
    const ::phi::CPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<float> value);

::phi::DenseTensor CreateGPUDenseTensor(
    const ::phi::GPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<::infrt::PrecisionType> precision);

void FillDenseTensorF32(::phi::DenseTensor* dense_tensor,
                        host_context::Attribute<std::vector<float>> values);
void PrintDenseTensor(::phi::DenseTensor* dense_tensor);

infrt::phi::DenseTensorMap LoadParams(
    host_context::Attribute<std::string> path);

::phi::DenseTensor TensorMapGetTensor(
    const ::infrt::phi::DenseTensorMap& map,
    host_context::Attribute<std::string> name);

::infrt::phi::DenseTensorMap LoadCombinedParams(
    host_context::Attribute<std::string> model_path,
    host_context::Attribute<std::string> params_path);

int32_t TensorMapGetSize(const ::infrt::phi::DenseTensorMap& map);

#ifdef INFRT_WITH_GPU
::phi::DenseTensor GpuMemCpy(const ::phi::DenseTensor& input,
                             const ::phi::GPUContext& context,
                             bool d2h);
#endif

}  // namespace phi
}  // namespace kernel
}  // namespace infrt
