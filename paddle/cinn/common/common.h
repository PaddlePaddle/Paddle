// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/strings/string_view.h>

#include "paddle/cinn/common/axis.h"
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"

namespace cinn {

// export some general concepts.
using cinn::common::Context;
using cinn::common::make_shared;
using cinn::common::Object;
using cinn::common::ref_count;
using cinn::common::Shared;
using cinn::common::UniqName;

// Type related.
using cinn::common::Bool;
using cinn::common::Float;
using cinn::common::Int;
using cinn::common::UInt;
using cinn::common::Void;

using cinn::common::type_of;

using cinn::common::Target;
using cinn::common::Type;
using cinn::common::UnkTarget;

template <typename T>
T& Reference(const T* x) {
  return *const_cast<T*>(x);
}

static void CheckVarNameValid(const absl::string_view name) {
  CHECK(!name.empty());
  CHECK(name.find(' ') == std::string::npos &&   //
        name.find('.') == std::string::npos &&   //
        name.find('@') == std::string::npos &&   //
        name.find('/') == std::string::npos &&   //
        name.find('\t') == std::string::npos &&  //
        name.find('\n') == std::string::npos &&  //
        name.find('\r') == std::string::npos)
      << "Some invalid character found";
  CHECK(!cinn::common::IsAxisNameReserved(std::string(name)))
      << "The name [" << name << "] is reserved for internal axis";
}

}  // namespace cinn
