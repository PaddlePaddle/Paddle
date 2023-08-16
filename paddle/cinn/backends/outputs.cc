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

#include "paddle/cinn/backends/outputs.h"

namespace cinn {
namespace lang {}  // namespace lang

backends::Outputs backends::Outputs::object(const std::string &name) const {
  Outputs updated = *this;
  updated.object_name = name;
  return updated;
}

backends::Outputs backends::Outputs::bitcode(const std::string &name) const {
  Outputs updated = *this;
  updated.bitcode_name = name;
  return updated;
}

backends::Outputs backends::Outputs::c_header(const std::string &name) const {
  Outputs updated = *this;
  updated.c_header_name = name;
  return updated;
}

backends::Outputs backends::Outputs::c_source(const std::string &name) const {
  Outputs updated = *this;
  updated.c_source_name = name;
  return updated;
}

backends::Outputs backends::Outputs::cuda_source(
    const std::string &name) const {
  Outputs updated = *this;
  updated.cuda_source_name = name;
  return updated;
}

}  // namespace cinn
