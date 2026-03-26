// Copyright (c) 2026 CINN Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/cinn/common/target.h"

namespace cinn {
namespace backends {
namespace cdrtc {

/**
 * An helper class to call Cdrtc or Cdcc. Input CUSTOMDEVICE device source code,
 * get hsaco string.
 */
class Compiler {
 public:
  explicit Compiler(const common::Target& target);
  /**
   * Compile the \p code and get hsaco string.
   * @param code The CUSTOMDEVICE source code.
   * @param include_headers Whether to include the headers of CUSTOMDEVICE and
   * CINN runtime modules.
   * @return Compiled hsaco code string.
   */
  std::string operator()(const std::string& code, bool include_headers = true);

 private:
  /**
   * The target is used to identify the specific Place/Device for
   * retrieving the corresponding plugin.
   */
  common::Target target_;
};

}  // namespace cdrtc
}  // namespace backends
}  // namespace cinn
