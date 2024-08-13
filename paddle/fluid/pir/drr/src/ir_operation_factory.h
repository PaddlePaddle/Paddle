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

#include <unordered_map>

#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/src/match_context_impl.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace paddle {
namespace drr {

class OperationFactory {
 public:
  static OperationFactory& Instance() {
    static OperationFactory operation_factory;
    return operation_factory;
  }

  using OperationCreateFunction =
      std::function<pir::Operation*(const std::vector<pir::Value>&,
                                    const pir::AttributeMap&,
                                    pir::PatternRewriter&)>;

  void RegisterOperationCreator(const std::string& op_name,
                                const OperationCreateFunction& create_fn) {
    op_creator_map[op_name] = create_fn;
  }

  pir::Operation* CreateOperation(
      const std::string& op_name,
      const std::vector<pir::Value>& inputs,
      const pir::AttributeMap& attrs,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    auto iter = op_creator_map.find(op_name);
    PADDLE_ENFORCE_NE(
        iter,
        op_creator_map.end(),
        common::errors::NotFound(
            "The op to be created is not found."
            "Suggest fix: Place check if the op named %s has been registered.",
            op_name));
    return iter->second(inputs, attrs, rewriter);
  }

 private:
  OperationFactory() {
    RegisterPdOpGeneratedOpCreator();
#ifdef PADDLE_WITH_CINN
    RegisterCinnOpGeneratedOpCreator();
#endif
#ifdef PADDLE_WITH_DNNL
    RegisterOnednnOpGeneratedOpCreator();
#endif
    RegisterManualOpCreator();
  }

  void RegisterManualOpCreator();
  void RegisterPdOpGeneratedOpCreator();
#ifdef PADDLE_WITH_CINN
  void RegisterCinnOpGeneratedOpCreator();
#endif
#ifdef PADDLE_WITH_DNNL
  void RegisterOnednnOpGeneratedOpCreator();
#endif
  std::unordered_map<std::string, OperationCreateFunction> op_creator_map;
};

pir::Operation* CreateOperation(const OpCall& op_call,
                                const MatchContextImpl& src_match_ctx,
                                pir::PatternRewriter& rewriter,  // NOLINT
                                MatchContextImpl* res_match_ctx);

}  // namespace drr
}  // namespace paddle
