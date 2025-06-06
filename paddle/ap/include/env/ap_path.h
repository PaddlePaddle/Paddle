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

#include <cstdlib>
#include "paddle/ap/include/adt/adt.h"

namespace ap::env {

template <typename YieldT>
adt::Result<adt::Ok> VisitEachApPath(const YieldT& Yield) {
  const char* ap_path_chars = std::getenv("AP_PATH");
  if (ap_path_chars == nullptr) {
    return adt::Ok{};
  }
  std::string ap_path(ap_path_chars);
  std::string path;
  std::istringstream ss(ap_path);
  while (std::getline(ss, path, ':')) {
    if (!path.empty()) {
      ADT_LET_CONST_REF(loop_ctr, Yield(path));
      if (loop_ctr.template Has<adt::Break>()) {
        break;
      }
    }
  }
  return adt::Ok{};
}

}  // namespace ap::env
