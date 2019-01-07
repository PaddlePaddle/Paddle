/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <sstream>

#include "paddle/fluid/platform/debug_support.h"

namespace paddle {
namespace platform {

template <>
std::string PythonDebugSupport::format() const {
  std::ostringstream sout;
  sout << "\nPython Callstacks: \n";
  for (auto& line : info) {
    sout << line;
  }
  return sout.str();
}

}  // namespace platform
}  // namespace paddle
