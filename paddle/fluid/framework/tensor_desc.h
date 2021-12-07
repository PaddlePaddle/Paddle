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
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/type_defs.h"

#include "boost/variant/get.hpp"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {

template <typename T>
const T& ExtractTensorDescValue(const TensorDescValue& desc_val) {
  // PADDLE_ENFORCE_EQ(desc_val.type(), typeid(std::vector<T>),
  //                   platform::errors::PreconditionNotMet(
  //                       "Found mismatch type in ExtractTensorDescValue."));
  return BOOST_GET_CONST(T, desc_val);
}

TensorDescValue GetTensorDescValue(
    const proto::VarType::TensorDesc& tensor_desc);

void SetTensorDescValue(proto::VarType::TensorDesc* tensor_desc,
                        const TensorDescValue& val);

}  // namespace framework
}  // namespace paddle
