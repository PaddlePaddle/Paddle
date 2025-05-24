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
#include <fstream>
#include <mutex>
#include <sstream>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/module_mgr.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/env/ap_path.h"
#include "paddle/ap/include/fs/fs.h"

namespace ap::registry {

struct RegistryMgr {
  static RegistryMgr* Singleton() {
    static RegistryMgr mgr{};
    return &mgr;
  }

  adt::Result<adt::Ok> LoadAllOnce() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!first_load_result_.has_value()) {
      using Ok = adt::Result<adt::Ok>;
      ADT_RETURN_IF_ERR(VisitApEntryFilePath([&](const auto& filepath) -> Ok {
        const Ok& cur_result = Load(filepath);
        if (!first_load_result_.has_value() && cur_result.HasError()) {
          first_load_result_ = cur_result;
        }
        return adt::Ok{};
      }));
      if (!first_load_result_.has_value()) {
        first_load_result_ = adt::Ok{};
      }
    }
    return first_load_result_.value();
  }

 private:
  std::optional<adt::Result<adt::Ok>> first_load_result_;
  std::mutex mutex_;

  adt::Result<adt::Ok> Load(const std::string& filepath) {
    static axpr::Lambda<axpr::CoreExpr> import([] {
      ap::axpr::LambdaExprBuilder lmd{};
      const ap::axpr::AnfExpr anf_expr =
          lmd.Lambda({"filepath"}, [&](auto& ctx) {
            ctx.Var("__builtin__import").Call(ctx.None(), ctx.Var("filepath"));
            return ctx.None();
          });
      const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
      const auto& atomic =
          core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
      return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
    }());

    ap::memory::Guard guard{};
    ap::axpr::Interpreter interpreter(
        axpr::MakeBuiltinFrameAttrMap<axpr::Value>(),
        guard.circlable_ref_list());
    ADT_RETURN_IF_ERR(
        interpreter.Interpret(import, std::vector<axpr::Value>{filepath}));
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitApEntryFilePath(const YieldT& Yield) {
    using Ctrl = adt::Result<adt::LoopCtrl>;
    ADT_RETURN_IF_ERR(env::VisitEachApPath([&](const auto& dir_path) -> Ctrl {
      const std::string file_path = std::string(dir_path) + "/__main__.py.json";
      if (fs::FileExists(file_path)) {
        ADT_RETURN_IF_ERR(Yield(file_path));
      }
      return adt::Continue{};
    }));
    return adt::Ok{};
  }
};

}  // namespace ap::registry
