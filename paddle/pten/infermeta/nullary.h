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

#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

// Common InferMeta Functions for 0-nary operators(no input tensor), The format
// like:
//
//   1. DenseTensorMeta [OpName]InferMeta( ...)
//  NOTE: The name "InferMeta" may be not appropriate. "InferMeta" may be good.
//  Because functions in this file
//  not only can infer shape, but alse need infer lod or other useful data.

DenseTensorMeta CreateInferMeta(const std::vector<int64_t>& shape,
                                DataType dtype,
                                DataLayout layout);

DenseTensorMeta CreateInferMeta(const ScalarArray& shape,
                                DataType dtype,
                                DataLayout layout);

}  // namespace pten
