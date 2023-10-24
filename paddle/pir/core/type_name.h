// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <string>

namespace pir {

template <typename DesiredTypeName>
inline std::string get_type_name() {
#if defined(__clang__) || defined(__GNUC__)
  std::string name = __PRETTY_FUNCTION__;
  std::string key = "DesiredTypeName = ";
  name = name.substr(name.find(key));
  assert(!name.empty() && "Unable to find the template parameter!");
  name = name.substr(key.size());
  assert(name.back() == ']' && "Name doesn't end in the substitution key!");
  auto sem_pos = name.find_first_of(";");
  if (sem_pos == std::string::npos)
    name.pop_back();
  else
    name = name.substr(0, sem_pos);
  return name;
#elif defined(_MSC_VER)
  std::string name = __FUNCSIG__;
  std::string key = "get_type_name<";
  name = name.substr(name.find(key));
  assert(!name.empty() && "Unable to find the function name!");
  name = name.substr(key.size());
  for (std::string prefix : {"class ", "struct ", "union ", "enum "}) {
    if (name.find(prefix) == 0) {
      name = name.substr(prefix.size());
      break;
    }
  }
  auto angle_pos = name.rfind('>');
  assert(angle_pos != std::string::npos && "Unable to find the closing '>'!");
  return name.substr(0, angle_pos);
#else
  // No known technique for statically extracting a type name on this compiler.
  // We return a string that is unlikely to look like any type in LLVM.
  return "UNKNOWN_TYPE";
#endif
}

}  // namespace pir
