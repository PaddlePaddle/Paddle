/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

class RenameGuard {
 public:
  RenameGuard(const Scope& scope,
              const std::vector<std::pair<std::string, std::string>>& names)
      : scope_(scope), names_(names) {
    ExchangeVarNames(scope, names);
  }

  ~RenameGuard() { ExchangeVarNames(scope_, names_); }

 private:
  void ExchangeVarNames(
      const Scope& scope,
      const std::vector<std::pair<std::string, std::string>>& names) {
    for (auto& name_pair : names) {
      auto temp_name = name_pair.first + name_pair.second;
      scope.Rename(name_pair.first, temp_name);
      scope.Rename(name_pair.second, name_pair.first);
      scope.Rename(temp_name, name_pair.second);
    }
  }

  const Scope& scope_;
  const std::vector<std::pair<std::string, std::string>>& names_;
};

}  // namespace framework
}  // namespace paddle
