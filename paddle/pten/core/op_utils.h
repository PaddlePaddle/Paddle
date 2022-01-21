/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <mutex>

#include "paddle/pten/core/arg_map_context.h"
#include "paddle/pten/core/infermeta_utils.h"
#include "paddle/pten/core/kernel_def.h"
#include "paddle/pten/core/macros.h"
#include "paddle/utils/flat_hash_map.h"

namespace pten {

class DefaultKernelSignatureMap {
 public:
  static DefaultKernelSignatureMap& Instance();

  bool Has(const std::string& op_type) const;

  const KernelSignature& Get(const std::string& op_type) const;

  void Insert(std::string op_type, KernelSignature signature);

 private:
  DefaultKernelSignatureMap() = default;

  paddle::flat_hash_map<std::string, KernelSignature> map_;

  DISABLE_COPY_AND_ASSIGN(DefaultKernelSignatureMap);
};

struct OpUtils {
  std::string api_name;
  ArgumentMappingFn arg_mapping_fn;

  OpUtils() {
    arg_mapping_fn = [&](const ArgumentMappingContext& ctx) -> KernelSignature {
      return DefaultKernelSignatureMap::Instance().Get(this->api_name);
    };
  }
};
class OpUtilsMap {
 public:
  static OpUtilsMap& Instance();

  bool Contains(const std::string& op_type) const;

  void InsertApiName(std::string op_type, std::string api_name);
  void InsertArgumentMappingFn(std::string op_type, ArgumentMappingFn fn);

  std::string GetApiName(const std::string& op_type) const;
  ArgumentMappingFn GetArgumentMappingFn(const std::string& op_type) const;

 private:
  OpUtilsMap() = default;

  paddle::flat_hash_map<std::string, std::string> name_map_;
  paddle::flat_hash_map<std::string, ArgumentMappingFn> arg_mapping_fn_map_;

  DISABLE_COPY_AND_ASSIGN(OpUtilsMap);
};

struct ApiNameRegistrar {
  ApiNameRegistrar(const char* op_type, const char* api_name) {
    OpUtilsMap::Instance().InsertApiName(op_type, api_name);
  }
};

struct ArgumentMappingFnRegistrar {
  ArgumentMappingFnRegistrar(const char* op_type,
                             ArgumentMappingFn arg_mapping_fn) {
    OpUtilsMap::Instance().InsertArgumentMappingFn(op_type,
                                                   std::move(arg_mapping_fn));
  }
};

#define PT_REGISTER_API_NAME(op_type, api_name)                             \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                        \
      pt_register_api_name_ns_check_##op_type,                              \
      "PT_REGISTER_API_NAME must be called in global namespace.");          \
  static const ::pten::ApiNameRegistrar __registrar_api_name_for_##op_type( \
      #op_type, #api_name);                                                 \
  int TouchApiNameSymbol_##op_type() { return 0; }

#define PT_DECLARE_API_NAME(op_type)                              \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                              \
      pt_declare_ai_name_ns_check_##op_type,                      \
      "PT_DECLARE_API_NAME must be called in global namespace."); \
  extern int TouchApiNameSymbol_##op_type();                      \
  UNUSED static int __declare_api_name_symbol_for_##op_type =     \
      TouchApiNameSymbol_##op_type()

#define PT_REGISTER_ARG_MAPPING_FN(op_type, arg_mapping_fn)              \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                     \
      pt_register_arg_map_fn_ns_check_##op_type,                         \
      "PT_REGISTER_ARG_MAPPING_FN must be called in global namespace."); \
  static const ::pten::ArgumentMappingFnRegistrar                        \
      __registrar_arg_map_fn_for_##op_type(#op_type, arg_mapping_fn);    \
  int TouchArgumentMappingFnSymbol_##op_type() { return 0; }

#define PT_DECLARE_ARG_MAPPING_FN(op_type)                              \
  PT_STATIC_ASSERT_GLOBAL_NAMESPACE(                                    \
      pt_declare_arg_map_fn_ns_check_##op_type,                         \
      "PT_DECLARE_ARG_MAPPING_FN must be called in global namespace."); \
  extern int TouchArgumentMappingFnSymbol_##op_type();                  \
  UNUSED static int __declare_arg_map_fn_symbol_for_##op_type =         \
      TouchArgumentMappingFnSymbol_##op_type()

}  // namespace pten
