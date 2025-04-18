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
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/dim_expr_method_class.h"
#include "paddle/ap/include/code_gen/code_gen_result_method_class.h"
#include "paddle/ap/include/code_module/code_module_method_class.h"
#include "paddle/ap/include/code_module/func_declare_method_class.h"
#include "paddle/ap/include/code_module/package_method_class.h"
#include "paddle/ap/include/code_module/project_method_class.h"

namespace ap::code_gen {

template <typename ValueT, typename DoEachT>
void VisitEachBuiltinFrameClass(const DoEachT& DoEach) {
  DoEach(code_module::GetProjectClass());
  DoEach(code_module::GetPackageClass());
  DoEach(code_module::GetFuncDeclareClass());
  DoEach(code_module::GetCodeModuleClass());
  DoEach(axpr::GetDimExprClass<ValueT>());
  DoEach(GetCodeGenResultClass());
}

template <typename ValueT>
axpr::AttrMap<ValueT> MakeBuiltinFrameAttrMap() {
  axpr::AttrMap<ValueT> attr_map;
  axpr::VisitEachBuiltinFrameAttr<ValueT>(
      [&](const std::string& k, const ValueT& v) { attr_map->Set(k, v); });
  VisitEachBuiltinFrameClass<ValueT>([&](const auto& cls) {
    attr_map->Set(cls.Name(), cls);
    attr_map->Set(std::string("__builtin__") + cls.Name(), cls);
  });
  return attr_map;
}

}  // namespace ap::code_gen
