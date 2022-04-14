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
#include "paddle/phi/common/data_type.h"

// type declaration forward
struct cinn_type_t;
namespace cinn::common {
struct Type;
}  // ::cinn::common

namespace paddle::framework::paddle2cinn {

::phi::DataType TransToPaddleDataType(const ::cinn::common::Type& type);

::phi::DataType TransToPaddleDataType(const cinn_type_t& type);

}  // namespace paddle::framework::paddle2cinn
