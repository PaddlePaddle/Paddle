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

#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/dense_tensor.h"

// This header is used to cast kernel function from void* to original form of
// function Currnetly.
// It may be generated automatically in the future.

namespace pten {

using DeviceContext = paddle::platform::DeviceContext;

using add_kernel = void (*)(const DeviceContext&,
                            const DenseTensor&,
                            const DenseTensor&,
                            int,
                            DenseTensor*);

using cast_kernel = void (*)(const DeviceContext&,
                             const DenseTensor&,
                             DataType,
                             DenseTensor*);

using divide_kernel = void (*)(const DeviceContext&,
                               const DenseTensor&,
                               const DenseTensor&,
                               int,
                               DenseTensor*);

using dot_kernel = void (*)(const DeviceContext&,
                            const DenseTensor&,
                            const DenseTensor&,
                            DenseTensor*);

using flatten_kernel =
    void (*)(const DeviceContext&, const DenseTensor&, int, int, DenseTensor*);

using empty_kernel = void (*)(const DeviceContext&,
                              const ScalarArray&,
                              DenseTensor*);

using empty_like_kernel = void (*)(const DeviceContext&, DenseTensor*);
using full_kernel = void (*)(const DeviceContext&,
                             const ScalarArray&,
                             const Scalar&,
                             DenseTensor*);

using full_like_kernel = void (*)(const DeviceContext&,
                                  const Scalar&,
                                  DenseTensor*);

using matmul_kernel = void (*)(const DeviceContext&,
                               const DenseTensor&,
                               const DenseTensor&,
                               bool,
                               bool,
                               DenseTensor*);

using mean_kernel = void (*)(const DeviceContext&,
                             const DenseTensor&,
                             const std::vector<int64_t>&,
                             bool,
                             bool,
                             DenseTensor*);

using multiply_kernel = void (*)(const DeviceContext&,
                                 const DenseTensor&,
                                 const DenseTensor&,
                                 int,
                                 DenseTensor*);

using reshape_kernel = void (*)(const DeviceContext&,
                                const DenseTensor&,
                                const ScalarArray&,
                                DenseTensor*);

using scale_kernel = void (*)(const DeviceContext&,
                              const DenseTensor&,
                              const Scalar&,
                              float,
                              bool,
                              DenseTensor*);

using sum_kernel = void (*)(const DeviceContext&,
                            const DenseTensor&,
                            const std::vector<int64_t>&,
                            bool,
                            bool,
                            DataType,
                            DenseTensor*);

using subtract_kernel = void (*)(const DeviceContext&,
                                 const DenseTensor&,
                                 const DenseTensor&,
                                 int,
                                 DenseTensor*);

using conj_kernel = void (*)(const DeviceContext&,
                             const DenseTensor&,
                             DenseTensor*);

}  // namespace pten
