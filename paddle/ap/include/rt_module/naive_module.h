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
#include "paddle/ap/include/code_module/func_declare.h"
#include "paddle/ap/include/rt_module/dl_handle.h"
#include "paddle/ap/include/rt_module/module.h"

namespace ap::rt_module {

class NaiveModule : public Module {
 public:
  NaiveModule(const NaiveModule&) = delete;
  NaiveModule(NaiveModule&&) = delete;

  adt::Result<Function> Get(const std::string& func_name) const override {
    const auto& iter = name2func_declare_.find(func_name);
    ADT_CHECK(iter != name2func_declare_.end()) << adt::errors::KeyError{
        std::string() + "function " + func_name + " is not declared"};
    const auto& func_declare = iter->second;
    ADT_LET_CONST_REF(dl_function, dl_handle_->DlSym(func_name));
    return Function{func_declare, dl_function};
  }

  static adt::Result<std::shared_ptr<const Module>> Make(
      const std::vector<code_module::FuncDeclare>& func_declares,
      const std::shared_ptr<const DlHandle>& dl_handle) {
    std::map<std::string, code_module::FuncDeclare> name2func_declare{};
    for (const auto& func_declare : func_declares) {
      ADT_CHECK(
          name2func_declare.emplace(func_declare->func_id, func_declare).second)
          << adt::errors::KeyError{
                 std::string() +
                 "duplicated function name: " + func_declare->func_id};
    }
    std::shared_ptr<const Module> m(
        new NaiveModule{name2func_declare, dl_handle});
    return m;
  }

 private:
  NaiveModule(
      const std::map<std::string, code_module::FuncDeclare>& name2func_declare,
      const std::shared_ptr<const DlHandle>& dl_handle)
      : name2func_declare_(name2func_declare), dl_handle_(dl_handle) {}
  std::map<std::string, code_module::FuncDeclare> name2func_declare_;
  std::shared_ptr<const DlHandle> dl_handle_;
};

}  // namespace ap::rt_module
