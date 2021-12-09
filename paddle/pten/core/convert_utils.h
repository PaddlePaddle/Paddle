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

#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/core/tensor_meta.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/place.h"

// TODO(chenweihang): this file may need to be removed

namespace pten {

using DataType = paddle::experimental::DataType;
using DataLayout = paddle::experimental::DataLayout;

const std::string& TransToPtenKernelName(const std::string& fluid_op_name);

Backend TransToPtenBackend(const paddle::platform::Place& place);
DataType TransToPtenDataType(
    const paddle::framework::proto::VarType::Type& dtype);
DataLayout TransToPtenDataLayout(const paddle::framework::DataLayout& layout);

paddle::platform::Place TransToFluidPlace(const Backend& backend);
paddle::framework::proto::VarType::Type TransToProtoVarType(
    const DataType& dtype);
paddle::framework::DataLayout TransToFluidDataLayout(const DataLayout& layout);

paddle::framework::LoD TransToFluidLoD(const pten::LoD& lod);
pten::LoD TransToPtenLoD(const paddle::framework::LoD& lod);

size_t DataTypeSize(DataType dtype);
DataType String2DataType(const std::string& str);
std::string DataType2String(DataType dtype);

}  // namespace pten
