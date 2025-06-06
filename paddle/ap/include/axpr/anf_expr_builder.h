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

#include "paddle/ap/include/axpr/anf_expr.h"
#include "paddle/ap/include/axpr/atomic_builder.h"

namespace ap::axpr {

class AnfExprBuilder : public AtomicExprBuilder<AnfExpr> {
 public:
  AnfExprBuilder() {}
  AnfExprBuilder(const AnfExprBuilder&) = delete;
  AnfExprBuilder(AnfExprBuilder&&) = delete;

  Combined<AnfExpr> Call(const Atomic<AnfExpr>& f,
                         const std::vector<Atomic<AnfExpr>>& args) {
    return Combined<AnfExpr>{ap::axpr::Call<AnfExpr>{f, args}};
  }

  Combined<AnfExpr> If(const Atomic<AnfExpr>& c,
                       const AnfExpr& t,
                       const AnfExpr& f) {
    return Combined<AnfExpr>{ap::axpr::If<AnfExpr>{c, t, f}};
  }

  ap::axpr::Bind<AnfExpr> Bind(const std::string& var,
                               const Combined<AnfExpr>& val) {
    return ap::axpr::Bind<AnfExpr>{tVar<std::string>{var}, val};
  }

  ap::axpr::Let<AnfExpr> Let(
      const std::vector<ap::axpr::Bind<AnfExpr>>& assigns,
      const AnfExpr& body) {
    return ap::axpr::Let<AnfExpr>{assigns, body};
  }

  AnfExpr operator()(const Atomic<AnfExpr>& atomic) { return AnfExpr{atomic}; }

  AnfExpr operator()(const Combined<AnfExpr>& combined) {
    return AnfExpr{combined};
  }

  AnfExpr operator()(const ap::axpr::Let<AnfExpr>& let) { return AnfExpr{let}; }

 private:
};

}  // namespace ap::axpr
