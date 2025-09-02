// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <ATen/core/TensorBody.h>

namespace at::detail {

using at::Tensor;
at::Tensor empty_cuda(IntArrayRef size,
                      ScalarType dtype,
                      std::optional<Device> device_opt,
                      std::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_cuda(IntArrayRef size, const TensorOptions &options);

}  // namespace at::detail
