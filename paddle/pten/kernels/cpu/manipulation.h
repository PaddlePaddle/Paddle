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

#include "paddle/pten/core/dense_tensor.h"
// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pten {

using CPUContext = paddle::platform::CPUDeviceContext;

template <typename T>
void Flatten(const CPUContext& dev_ctx,
             const DenseTensor& x,
             int start_axis,
             int stop_axis,
             DenseTensor* out);

void ReshapeFromDT(const CPUContext& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& shape,
                   DenseTensor* out);

void ReshapeFromVectorVal(const CPUContext& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<int64_t>& shape,
                          DenseTensor* out);

void ReshapeFromVectorDT(const CPUContext& dev_ctx,
                         const DenseTensor& x,
                         const std::vector<DenseTensor>& shape,
                         DenseTensor* out);

void ReshapeFromDTWithXShape(const CPUContext& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& shape,
                             DenseTensor* xshape,
                             DenseTensor* out);

void ReshapeFromVectorValWithXShape(const CPUContext& dev_ctx,
                                    const DenseTensor& x,
                                    const std::vector<int64_t>& shape,
                                    DenseTensor* xshape,
                                    DenseTensor* out);

void ReshapeFromVectorDTWithXShape(const CPUContext& dev_ctx,
                                   const DenseTensor& x,
                                   const std::vector<DenseTensor>& shape,
                                   DenseTensor* xshape,
                                   DenseTensor* out);

}  // namespace pten
