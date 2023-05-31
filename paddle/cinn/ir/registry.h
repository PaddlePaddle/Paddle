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

#include <string>
#include <vector>

#include "paddle/cinn/lang/packed_func.h"

namespace cinn::ir {

class Registry {
 public:
  Registry &SetBody(lang::PackedFunc f);
  Registry &SetBody(lang::PackedFunc::body_t f);

  static Registry &Register(const std::string &name, bool can_override = false);
  static bool Remove(const std::string &name);
  static const lang::PackedFunc *Get(const std::string &name);
  static std::vector<std::string> ListNames();

  struct Manager;

  explicit Registry(const std::string &);

 protected:
  std::string name_;
  lang::PackedFunc func_;
  friend class Manager;
};

}  // namespace cinn::ir
