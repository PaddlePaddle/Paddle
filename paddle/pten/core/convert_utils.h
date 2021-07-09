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

#include "paddle/pten/core/backend.h"
#include "paddle/pten/core/dtype.h"
#include "paddle/pten/core/layout.h"

// fluid headers [may be replaced by new impl]
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/place.h"

// TODO(chenweihang): this file may need to be removed

namespace pt {

// TODO(chenweihang): Use the original var type as much as possible
// to avoid transform, such as DataLayout, VarType
Backend TransToPtenBackend(const paddle::platform::Place& place);
DataType TransToPtenDataType(
    const paddle::framework::proto::VarType::Type& dtype);
Layout TransToPtenLayout(const paddle::framework::DataLayout& layout);
paddle::framework::proto::VarType::Type TransToProtoVarType(
    const DataType& dtype);

}  // namespace pt
