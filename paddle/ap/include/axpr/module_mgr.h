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
#include "glog/logging.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/builtin_func_name_mgr.h"
#include "paddle/ap/include/axpr/frame.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/env/ap_path.h"
#include "paddle/ap/include/memory/guard.h"
#include "paddle/ap/include/preprocessor/preprocessor.h"

namespace ap::axpr {

class ModuleMgr {
 public:
  ModuleMgr()
      : memory_guard_(),
        file_path2const_global_frame_(),
        module_name2const_global_frame_() {}

  ModuleMgr(const ModuleMgr&) = delete;
  ModuleMgr(ModuleMgr&&) = delete;

  static ModuleMgr* Singleton() {
    static ModuleMgr module_mgr;
    return &module_mgr;
  }

  std::optional<axpr::AttrMap<SerializableValue>> OptGetBuiltinModule(
      const std::string& module_name) {
    const auto iter = module_name2builtin_module_.find(module_name);
    if (iter == module_name2builtin_module_.end()) return std::nullopt;
    return iter->second;
  }

  template <typename InitT>
  adt::Result<Frame<SerializableValue>> GetOrCreateByModuleName(
      const std::string& module_name, const InitT& Init) {
    {
      const auto& iter = module_name2const_global_frame_.find(module_name);
      if (iter != module_name2const_global_frame_.end()) {
        return iter->second;
      }
    }
    ADT_LET_CONST_REF(file_path, GetFilePathByModuleName(module_name))
        << adt::errors::ModuleNotFoundError{
               std::string() + "No module named '" + module_name + "'"};
    ADT_LET_CONST_REF(frame, GetOrCreateByFilePath(file_path, Init));
    ADT_CHECK(
        module_name2const_global_frame_.emplace(module_name, frame).second);
    return frame;
  }

  template <typename InitT>
  adt::Result<Frame<SerializableValue>> GetOrCreateByFilePath(
      const std::string& file_path, const InitT& Init) {
    const auto& iter = file_path2const_global_frame_.find(file_path);
    if (iter != file_path2const_global_frame_.end()) {
      return iter->second;
    }
    auto frame_object = std::make_shared<AttrMapImpl<SerializableValue>>();
    frame_object->Set("__file__", file_path);
    const auto& frame =
        Frame<SerializableValue>::Make(circlable_ref_list(), frame_object);
    ADT_LET_CONST_REF(lambda, GetLambdaByFilePath(file_path));
    ADT_CHECK(file_path2const_global_frame_.emplace(file_path, frame).second);
    ADT_RETURN_IF_ERR(Init(frame, lambda));
    return frame;
  }

  const std::shared_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list()
      const {
    return memory_guard_.circlable_ref_list();
  }

  void RegisterBuiltinFrame(const std::string& name,
                            const axpr::AttrMap<SerializableValue>& attr_map) {
    CHECK(module_name2builtin_module_.emplace(name, attr_map).second);
  }

 private:
  adt::Result<std::string> GetFilePathByModuleName(
      const std::string& module_name) {
    std::optional<std::string> file_path;
    using RetT = adt::Result<adt::LoopCtrl>;
    ADT_RETURN_IF_ERR(
        VisitEachConfigFilePath([&](const std::string& dir_name) -> RetT {
          const std::string& cur_file_path =
              dir_name + "/" + module_name + ".py.json";
          if (FileExists(cur_file_path)) {
            file_path = cur_file_path;
            return adt::Break{};
          } else {
            return adt::Continue{};
          }
        }));
    ADT_CHECK(file_path.has_value());
    return file_path.value();
  }

  adt::Result<axpr::Lambda<axpr::CoreExpr>> GetLambdaByFilePath(
      const std::string& file_path) {
    ADT_LET_CONST_REF(file_content, GetFileContent(file_path));
    ADT_CHECK(!file_content.empty());
    ADT_LET_CONST_REF(anf_expr, axpr::MakeAnfExprFromJsonString(file_content));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    std::vector<axpr::tVar<std::string>> args{};
    axpr::Lambda<axpr::CoreExpr> lambda{args, core_expr};
    return lambda;
  }

  adt::Result<std::string> GetFileContent(const std::string& filepath) {
    std::ifstream ifs(filepath);
    std::string content{std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>()};
    return content;
  }

  bool FileExists(const std::string& filepath) {
    std::fstream fp;
    fp.open(filepath, std::fstream::in);
    if (fp.is_open()) {
      fp.close();
      return true;
    } else {
      return false;
    }
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitEachConfigFilePath(const DoEachT& DoEach) {
    return ap::env::VisitEachApPath(DoEach);
  }

  memory::Guard memory_guard_;

  std::unordered_map<std::string, Frame<SerializableValue>>
      file_path2const_global_frame_;

  std::unordered_map<std::string, Frame<SerializableValue>>
      module_name2const_global_frame_;

  std::unordered_map<std::string, axpr::AttrMap<SerializableValue>>
      module_name2builtin_module_;
};

struct ApBuiltinModuleBuilder {
  std::string module_name;
  axpr::AttrMap<SerializableValue> attr_map{};

  void Def(const std::string& name,
           const axpr::BuiltinFuncType<axpr::Value>& func) {
    void* func_ptr = reinterpret_cast<void*>(func);
    attr_map->Set(name, BuiltinFuncVoidPtr{func_ptr});
    BuiltinFuncNameMgr::Singleton()->Register(module_name, name, func_ptr);
  }

  void Def(const std::string& name,
           const axpr::BuiltinHighOrderFuncType<axpr::Value>& func) {
    void* func_ptr = reinterpret_cast<void*>(func);
    attr_map->Set(name, BuiltinHighOrderFuncVoidPtr{func_ptr});
    BuiltinFuncNameMgr::Singleton()->Register(module_name, name, func_ptr);
  }
};

struct ApBuiltinModuleRegistryHelper {
  ApBuiltinModuleRegistryHelper(
      const std::string& name,
      const std::function<void(ApBuiltinModuleBuilder*)>& func) {
    ApBuiltinModuleBuilder builder{name};
    func(&builder);
    ModuleMgr::Singleton()->RegisterBuiltinFrame(name, builder.attr_map);
  }
};

#define REGISTER_AP_BUILTIN_MODULE(name, ...)                          \
  namespace {                                                          \
  ::ap::axpr::ApBuiltinModuleRegistryHelper AP_CONCAT(                 \
      ap_builtin_module_registry_helper, __LINE__)(name, __VA_ARGS__); \
  }

}  // namespace ap::axpr
