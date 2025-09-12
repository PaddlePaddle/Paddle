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
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/module_mgr.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/env/ap_path.h"
#include "paddle/ap/include/fs/fs.h"
#include "paddle/ap/include/registry/builtin_frame_util.h"
#include "paddle/ap/include/registry/value.h"

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
    ADT_LET_CONST_REF(file_content, GetFileContent(filepath));
    if (file_content.empty()) {
      return adt::Ok{};
    }
    ADT_LET_CONST_REF(anf_expr, axpr::MakeAnfExprFromJsonString(file_content));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    const auto& frame = axpr::Frame<axpr::SerializableValue>::Make(
        axpr::ModuleMgr::Singleton()->circlable_ref_list(),
        std::make_shared<axpr::AttrMapImpl<axpr::SerializableValue>>());
    std::vector<axpr::tVar<std::string>> args{};
    axpr::Lambda<axpr::CoreExpr> lambda{args, core_expr};
    memory::Guard guard{};
    axpr::Interpreter cps_expr_interpreter(
        registry::MakeBuiltinFrameAttrMap<registry::Val>(),
        guard.circlable_ref_list());
    ADT_RETURN_IF_ERR(cps_expr_interpreter.InterpretModule(frame, lambda));
    return adt::Ok{};
  }

  adt::Result<std::string> GetFileContent(const std::string& filepath) {
    std::ifstream ifs(filepath);
    std::string content{std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>()};
    return content;
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
