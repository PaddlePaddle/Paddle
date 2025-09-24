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

#include "paddle/ap/include/axpr/s_expr.h"
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace ap::axpr {

namespace {

std::string AtomicExprToSExpression(const Atomic<SExpr>& s_expr) {
  return s_expr.Match(
      [](const tVar<std::string>& var) { return var.value(); },
      [](const adt::Nothing&) { return std::string("()"); },
      [](const bool c) { return c ? std::string("#t") : std::string("#f"); },
      [](const int64_t c) { return std::to_string(c); },
      [](const double c) { return std::to_string(c); },
      [](const std::string& str) {
        std::ostringstream ss;
        ss << std::quoted(str);
        return ss.str();
      },
      [](const Lambda<SExpr>& lambda) {
        std::ostringstream ss;
        ss << "(lambda [";
        int i = 0;
        for (const auto& arg : lambda->args) {
          if (i++ > 0) {
            ss << " ";
          }
          ss << arg.value();
        }
        ss << "] ";
        ss << lambda->body.ToSExpression();
        ss << ")";
        return ss.str();
      });
}

std::string SListExprToSExpression(const SList<SExpr>& s_expr) {
  std::ostringstream ss;
  int i = 0;
  ss << "(";
  for (const auto& child : s_expr->children) {
    if (i++ > 0) {
      ss << " ";
    }
    ss << child.ToSExpression();
  }
  ss << ")";
  return ss.str();
}

}  // namespace

std::string SExpr::ToSExpression() const {
  return Match(
      [&](const Atomic<SExpr>& s_expr) {
        return AtomicExprToSExpression(s_expr);
      },
      [&](const SList<SExpr>& s_expr) {
        return SListExprToSExpression(s_expr);
      });
}

}  // namespace ap::axpr
