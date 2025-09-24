// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/code_module/package.h"
#include "paddle/ap/include/code_module/project.h"

namespace ap::code_module {

using SourceCodeImpl = std::variant<Project, Package>;

struct SourceCode : public SourceCodeImpl {
  using SourceCodeImpl::SourceCodeImpl;
  ADT_DEFINE_VARIANT_METHODS(SourceCodeImpl);

  static adt::Result<SourceCode> CastFromAxprValue(const axpr::Value& val) {
    if (val.template CastableTo<Project>()) {
      ADT_LET_CONST_REF(project, val.template CastTo<Project>());
      return project;
    }
    if (val.template CastableTo<Package>()) {
      ADT_LET_CONST_REF(package, val.template CastTo<Package>());
      return package;
    }
    return adt::errors::TypeError{"SourceCode::CastFromAxprValue() failed"};
  }
};

}  // namespace ap::code_module
