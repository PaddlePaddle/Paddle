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

#include "paddle/ap/include/axpr/anf_expr.h"
#include "paddle/ap/include/axpr/anf_expr_builder.h"

#include <glog/logging.h>
#include <exception>
#include <unordered_map>
#include "nlohmann/json.hpp"

namespace ap::axpr {

using adt::Result;

using Json = nlohmann::json;

Json ConvertAnfExprToJson(const AnfExpr& anf_expr);

Json ConvertAtomicAnfExprToJson(const Atomic<AnfExpr>& atomic_expr) {
  return atomic_expr.Match(
      [&](const tVar<std::string>& var) {
        Json j = var.value();
        return j;
      },
      [&](adt::Nothing) {
        Json j;
        return j;
      },
      [&](bool c) {
        Json j = c;
        return j;
      },
      [&](int64_t c) {
        Json j = c;
        return j;
      },
      [&](double c) {
        Json j = c;
        return j;
      },
      [&](const std::string& c) {
        Json j;
        j[AnfExpr::kString()] = c;
        return j;
      },
      [&](const Lambda<AnfExpr>& lambda) {
        Json j = Json::array();
        j.push_back(AnfExpr::kLambda());
        j.push_back([&] {
          Json j_args = Json::array();
          for (const auto& arg : lambda->args) {
            j_args.push_back(arg.value());
          }
          return j_args;
        }());
        j.push_back(ConvertAnfExprToJson(lambda->body));
        return j;
      });
}

Json ConvertCombinedAnfExprToJson(const Combined<AnfExpr>& combined_expr) {
  return combined_expr.Match(
      [&](const Call<AnfExpr>& call_expr) {
        Json j;
        j.push_back(ConvertAtomicAnfExprToJson(call_expr->func));
        for (const auto& arg : call_expr->args) {
          j.push_back(ConvertAtomicAnfExprToJson(arg));
        }
        return j;
      },
      [&](const If<AnfExpr>& if_expr) {
        Json j;
        j.push_back(AnfExpr::kIf());
        j.push_back(ConvertAtomicAnfExprToJson(if_expr->cond));
        j.push_back(ConvertAnfExprToJson(if_expr->true_expr));
        j.push_back(ConvertAnfExprToJson(if_expr->false_expr));
        return j;
      });
}

Json ConvertBindingAnfExprToJson(const Bind<AnfExpr>& binding_expr) {
  Json j = Json::array();
  j.push_back(binding_expr.var.value());
  j.push_back(ConvertCombinedAnfExprToJson(binding_expr.val));
  return j;
}

Json ConvertLetAnfExprToJson(const Let<AnfExpr>& let_expr) {
  Json j;
  j.push_back(AnfExpr::kLet());
  Json j_array = Json::array();
  for (const auto& binding : let_expr->bindings) {
    j_array.push_back(ConvertBindingAnfExprToJson(binding));
  }
  j.push_back(j_array);
  j.push_back(ConvertAnfExprToJson(let_expr->body));
  return j;
}

Json ConvertAnfExprToJson(const AnfExpr& anf_expr) {
  return anf_expr.Match(
      [&](const Atomic<AnfExpr>& atomic_expr) {
        return ConvertAtomicAnfExprToJson(atomic_expr);
      },
      [&](const Combined<AnfExpr>& combined_expr) {
        return ConvertCombinedAnfExprToJson(combined_expr);
      },
      [&](const Let<AnfExpr>& let_expr) {
        return ConvertLetAnfExprToJson(let_expr);
      });
}

std::string AnfExpr::DumpToJsonString() const {
  return ConvertAnfExprToJson(*this).dump();
}

std::string AnfExpr::DumpToJsonString(int indent) const {
  return ConvertAnfExprToJson(*this).dump(indent);
}

}  // namespace ap::axpr
