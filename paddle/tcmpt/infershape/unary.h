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
#include "paddle/tcmpt/core/tensor_meta.h"

namespace pt {

// Common InferShape Functions, The format like:
//
//   1. TensorMeta [OpName]InferShape(const TensorMeta& x_meta, ...) {}
//   2. std::pair<TensorMeta, TensorMeta> [OpName]InferShape(const TensorMeta&
//   x_meta, ...) {}
//   3. std::tuple<TensorMeta, TensorMeta, TensorMeta> [OpName]InferShape(const
//   TensorMeta& x_meta, ...)
//  NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
//  Because functions in this file
//  not only can infer shape, but alse need infer lod or other useful data.

TensorMeta UnchangedInferShape(const TensorMeta& x_meta);

TensorMeta MeanInferShape(const TensorMeta& x_meta);

TensorMeta DotInferShape(const TensorMeta& x_meta);

TensorMeta FlattenInferShape(const TensorMeta& x_meta,
                             int start_axis,
                             int stop_axis);

}  // namespace pt
