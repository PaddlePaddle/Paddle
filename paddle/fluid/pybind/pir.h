// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <pybind11/pybind11.h>
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/pir/core/op_result.h"

namespace paddle {
namespace pybind {
using pir::OpResult;
void BindPir(pybind11::module *m);
phi::DataType GetOpResultDtype(const OpResult &result);
const phi::DDim &GetOpResultDims(const OpResult &result);
bool GetOpResultBoolAttr(const OpResult &self, const std::string &attr_name);
}  // namespace pybind
}  // namespace paddle
