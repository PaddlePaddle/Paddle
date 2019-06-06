// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file implements the interface to manipute the protobuf message. We use
 * macros to make a compatible interface with the framework::XXDesc and
 * lite::pb::XXDesc.
 */

#include "paddle/fluid/lite/core/framework.pb.h"
#include "paddle/fluid/lite/model_parser/cpp/op_desc.h"
#include "paddle/fluid/lite/model_parser/pb/op_desc.h"
#include "paddle/fluid/lite/model_parser/pb/var_desc.h"

namespace paddle {
namespace lite {

using Attribute = lite::pb::Attribute;
using OpDesc = lite::pb::OpDesc;
using VarDesc = lite::pb::VarDesc;

template <typename T>
T GetAttr(const Attribute& x) {
  return x.get<T>();
}

/// Transform an OpDesc from pb to cpp format.
void TransformOpDescPbToCpp(const pb::OpDesc& pb_desc, cpp::OpDesc* cpp_desc);

/// Transform an OpDesc from cpp to pb format.
void TransformOpDescCppToPb(const cpp::OpDesc& cpp_desc, pb::OpDesc* pb_desc);

}  // namespace lite
}  // namespace paddle
