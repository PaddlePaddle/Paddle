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
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/code_module/adt.h"
#include "paddle/ap/include/code_module/arg_type.h"
#include "paddle/ap/include/code_module/data_type.h"
#include "paddle/ap/include/code_module/file.h"

namespace ap::code_module {

struct PackageImpl {
  Directory<File> nested_files;
  std::string api_wrapper_so_relative_path;
  std::string main_so_relative_path;
  axpr::AttrMap<axpr::SerializableValue> others;

  bool operator==(const PackageImpl& other) const { return this == &other; }
};
ADT_DEFINE_RC(Package, PackageImpl);

}  // namespace ap::code_module
