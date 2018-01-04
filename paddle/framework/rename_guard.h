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
    for (auto& name_pair : names) {
      VLOG(3) << "ExchangeVarNames from " << name_pair.first << " to "
              << name_pair.second;
      if (scope.FindVarLocally(name_pair.first)) {
        auto temp_name = name_pair.first + name_pair.second;
        scope.Rename(name_pair.first, temp_name);
        scope.Rename(name_pair.second, name_pair.first);
        scope.Rename(temp_name, name_pair.second);
      } else {
        scope.Rename(name_pair.second, name_pair.first);
      }
    }
  }

  ~RenameGuard() {
    for (auto& name_pair : names_) {
      VLOG(3) << "ExchangeVarNames back from " << name_pair.first << " to "
              << name_pair.second;
      if (scope_.FindVarLocally(name_pair.second)) {
        auto temp_name = name_pair.first + name_pair.second;
        scope_.Rename(name_pair.first, temp_name);
        scope_.Rename(name_pair.second, name_pair.first);
        scope_.Rename(temp_name, name_pair.second);
      } else {
        scope_.Rename(name_pair.first, name_pair.second);
      }
    }
  }

 private:
  const Scope& scope_;
  const std::vector<std::pair<std::string, std::string>>& names_;
};

}  // namespace framework
}  // namespace paddle
