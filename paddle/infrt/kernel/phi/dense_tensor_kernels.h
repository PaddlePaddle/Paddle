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

<<<<<<< HEAD
::Tensor CreateDenseTensor(
=======
::phi::DenseTensor CreateDenseTensor(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const ::phi::CPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<::infrt::PrecisionType> precision);

<<<<<<< HEAD
::Tensor CreateInitedDenseTensorF32(
=======
::phi::DenseTensor CreateInitedDenseTensorF32(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const ::phi::CPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<float> value);

<<<<<<< HEAD
::Tensor CreateHostInitedDenseTensorF32(
=======
::phi::DenseTensor CreateHostInitedDenseTensorF32(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const ::phi::CPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<std::vector<float>> values);

<<<<<<< HEAD
::Tensor CreateGPUDenseTensor(
=======
::phi::DenseTensor CreateGPUDenseTensor(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const ::phi::GPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<::infrt::PrecisionType> precision);

<<<<<<< HEAD
void FillDenseTensorF32(::Tensor* dense_tensor,
                        host_context::Attribute<std::vector<float>> values);
void PrintDenseTensor(::Tensor* dense_tensor);
=======
void FillDenseTensorF32(::phi::DenseTensor* dense_tensor,
                        host_context::Attribute<std::vector<float>> values);
void PrintDenseTensor(::phi::DenseTensor* dense_tensor);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

::infrt::phi::DenseTensorMap LoadParameters(const std::string& path);

::infrt::phi::DenseTensorMap LoadParams(
    host_context::Attribute<std::string> path);

<<<<<<< HEAD
::Tensor TensorMapGetTensor(const ::infrt::phi::DenseTensorMap& map,
                            host_context::Attribute<std::string> name);
=======
::phi::DenseTensor TensorMapGetTensor(
    const ::infrt::phi::DenseTensorMap& map,
    host_context::Attribute<std::string> name);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

::infrt::phi::DenseTensorMap LoadCombinedParams(
    host_context::Attribute<std::string> model_path,
    host_context::Attribute<std::string> params_path);

::infrt::phi::DenseTensorMap LoadCombinedParameters(
    const std::string& model_path, const std::string& params_path);

::infrt::phi::DenseTensorMap LoadCombinedParamsToGpu(
    const std::string& model_path, const std::string& params_path);

int32_t TensorMapGetSize(const ::infrt::phi::DenseTensorMap& map);

#ifdef INFRT_WITH_GPU
<<<<<<< HEAD
void GpuMemCpy(const ::Tensor& input,
               const ::phi::GPUContext& context,
               bool d2h,
               ::Tensor* output);
=======
void GpuMemCpy(const ::phi::DenseTensor& input,
               const ::phi::GPUContext& context,
               bool d2h,
               ::phi::DenseTensor* output);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#endif

}  // namespace phi
}  // namespace kernel
}  // namespace infrt
