/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <unordered_map>
#include "Function.h"
#include "Tensor.h"
#include "meta/AttributeMeta.h"
#include "paddle/utils/Any.h"
namespace paddle {
namespace topology {

/**
 * @brief validate a function and inference the output shape.
 */
paddle::Error validateAndInferShape(Function& func);

/**
 * @brief validate a attribute map.
 * @return
 */
paddle::Error validate(const meta::AttributeMetaMap& meta, AttributeMap& attr);

}  // namespace topology
}  // namespace paddle
