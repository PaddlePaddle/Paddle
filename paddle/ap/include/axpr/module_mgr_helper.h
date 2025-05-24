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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/interpreter_base.h"
#include "paddle/ap/include/axpr/module_mgr.h"

namespace ap::axpr {

template <typename ValueT>
struct ModuleMgrHelper {
  using This = ModuleMgrHelper;

  static adt::Result<ValueT> ImportModule(InterpreterBase<ValueT>* interpreter,
                                          const ValueT&,
                                          const std::vector<ValueT>& args) {
    return This{}.Import(interpreter, args);
  }

  adt::Result<ValueT> Import(InterpreterBase<ValueT>* interpreter,
                             const std::vector<ValueT>& args) {
    std::optional<std::string> opt_module_name;
    std::optional<std::string> opt_module_filepath;
    ADT_CHECK(args.size() <= 2);
    if (args.size() >= 1) {
      if (args.at(0).template CastableTo<adt::Nothing>()) {
        opt_module_name = std::nullopt;
      } else {
        ADT_LET_CONST_REF(module_name,
                          args.at(0).template CastTo<std::string>());
        opt_module_name = module_name;
      }
    }
    if (args.size() >= 2) {
      ADT_LET_CONST_REF(module_filepath,
                        args.at(1).template CastTo<std::string>());
      opt_module_filepath = module_filepath;
    }
    ADT_CHECK(opt_module_name.has_value() || opt_module_filepath.has_value());
    auto* module_mgr = ModuleMgr::Singleton();
    if (opt_module_name.has_value()) {
      const auto& opt_builtin_module =
          module_mgr->OptGetBuiltinModule(opt_module_name.value());
      if (opt_builtin_module.has_value()) {
        return opt_builtin_module.value();
      }
    }
    auto Init = [&](const Frame<SerializableValue>& frame,
                    const axpr::Lambda<axpr::CoreExpr>& lambda)
        -> adt::Result<adt::Ok> {
      ADT_RETURN_IF_ERR(interpreter->InterpretModule(frame, lambda));
      return adt::Ok{};
    };
    ADT_LET_CONST_REF(
        frame,
        module_mgr->GetOrCreate(opt_module_name, opt_module_filepath, Init));
    ADT_LET_CONST_REF(frame_impl_obj, frame.shared_ptr());
    return axpr::AttrMap<SerializableValue>{frame_impl_obj};
  }
};

}  // namespace ap::axpr
