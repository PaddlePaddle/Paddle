// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "ATen/core/function_schema.h"

namespace c10 {

std::ostream& operator<<(std::ostream& out, const Argument& arg) {
  out << arg.type()->str() << " " << arg.name();
  if (arg.default_value()) {
    out << " = " << arg.default_value();
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  out << "(";
  bool first = true;
  for (const auto& arg : schema.arguments()) {
    if (!first) {
      out << ", ";
    }
    out << arg;
    first = false;
  }
  if (schema.is_vararg()) {
    if (!first) {
      out << ", ";
    }
    out << "...";
  }
  out << ")";

  out << " -> ";

  if (schema.returns().size() == 1) {
    out << schema.returns()[0];
  } else {
    out << "(";
    first = true;
    for (const auto& ret : schema.returns()) {
      if (!first) {
        out << ", ";
      }
      out << ret;
      first = false;
    }
    out << ")";
  }

  return out;
}

}  // namespace c10
