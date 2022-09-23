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

#include <string>

#include "paddle/infrt/common/macros.h"
#include "paddle/infrt/common/shared.h"
#include "paddle/infrt/common/target.h"
#include "paddle/infrt/common/type.h"

namespace infrt {

// export some general concepts.
using common::make_shared;
using common::Object;
using common::ref_count;
using common::Shared;

// Type related.
using common::Bool;
using common::Float;
using common::Int;
using common::UInt;
using common::Void;

using common::type_of;

using common::Target;
using common::Type;
using common::UnkTarget;

template <typename T>
T& Reference(const T* x) {
  return *const_cast<T*>(x);
}

static void CheckVarNameValid(const std::string& name) {
  CHECK(!name.empty());
  CHECK(name.find(' ') == std::string::npos &&   //
        name.find('.') == std::string::npos &&   //
        name.find('/') == std::string::npos &&   //
        name.find('\t') == std::string::npos &&  //
        name.find('\n') == std::string::npos &&  //
        name.find('\r') == std::string::npos)
      << "Some invalid character found";
}

}  // namespace infrt
