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

// See Note [ Why still include the fluid headers? ]
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

// Common InferMeta Functions for binary operators, The format like:
//
//   1. DenseTensorMeta [OpName]InferMeta(const DenseTensorMeta& x_meta, ...)
//   {}
//   2. std::pair<DenseTensorMeta, DenseTensorMeta> [OpName]InferMeta(const
//   DenseTensorMeta&
//   x_meta, ...) {}
//   3. std::tuple<DenseTensorMeta, DenseTensorMeta, DenseTensorMeta>
//   [OpName]InferMeta(const
//   DenseTensorMeta& x_meta, ...)
//  NOTE: The name "InferMeta" may be not appropriate. "InferMeta" may be good.
//  Because functions in this file
//  not only can infer shape, but alse need infer lod or other useful data.

DenseTensorMeta DotInferMeta(const DenseTensorMeta& x_meta,
                             const DenseTensorMeta& y_meta);

DenseTensorMeta MatmulInferMeta(const DenseTensorMeta& x_meta,
                                const DenseTensorMeta& y_meta,
                                bool trans_x,
                                bool trans_y);

DenseTensorMeta ElementwiseInferMeta(const DenseTensorMeta& x_meta,
                                     const DenseTensorMeta& y_meta,
                                     int axis);
}  // namespace pten
