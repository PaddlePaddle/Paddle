/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <string>
#include <utility>

#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/ddim.h"

namespace phi {

class RecordOpInfoSupplement {
 public:
  static bool IsEnabled();

  RecordOpInfoSupplement() = default;

  explicit RecordOpInfoSupplement(
      const std::string& type,
      const std::vector<std::pair<const char*, std::vector<DDim>>>&
          input_shapes,
      const AttributeMap& attrs);
};

}  // namespace phi
