// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/eager/legacy/type_def.h"
#include "paddle/fluid/imperative/jit/program_desc_tracer.h"
#include "paddle/pten/core/tensor_meta.h"

namespace egr {
namespace legacy {

void RunOp(const std::string& type, const NameTensorMap& ins,
           const NameTensorMap& outs, paddle::framework::AttributeMap attrs,
           const paddle::platform::Place& place,
           paddle::framework::AttributeMap* default_attrs,
           bool override_default_attr_map,
           const std::map<std::string, std::string>& inplace_map = {});

}  // namespace legacy
}  // namespace egr
