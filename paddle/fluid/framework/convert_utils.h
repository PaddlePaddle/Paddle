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

#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/tensor_meta.h"

#include "paddle/fluid/framework/data_type.h"

// TODO(chenweihang): this file may need to be removed

namespace paddle {
namespace framework {

using DataType = paddle::experimental::DataType;
using DataLayout = paddle::experimental::DataLayout;

DataType TransToPhiDataType(
    const paddle::framework::proto::VarType::Type& dtype);

paddle::framework::proto::VarType::Type TransToProtoVarType(
    const DataType& dtype);

size_t DataTypeSize(DataType dtype);
DataType String2DataType(const std::string& str);
std::string DataType2String(DataType dtype);

}  // namespace framework
}  // namespace paddle
