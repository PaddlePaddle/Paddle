/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/api/ext/op_meta_info.h"

namespace paddle {
namespace framework {

namespace detail {

// dynamic lib load func
template <typename T>
static T* DynLoad(void* handle, std::string name) {
  T* func = reinterpret_cast<T*>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func,
      platform::errors::NotFound(
          "Failed to load dynamic operator library, error message(%s).",
          errorno));
  return func;
}

inline static bool IsDuplicableVar(const std::string& var_name) {
  std::string suffix = kTensorVectorSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

inline static bool IsOptionalVar(const std::string& var_name) {
  std::string suffix = kOptionalSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

inline static std::string NoGrad(const std::string& var_name,
                                 bool is_double_grad = false) {
  std::string suffix = kGradVarSuffix;
  std::string new_out_suffix = kDoubleGradNewOutSuffix;
  std::string tmp_var_name(var_name);
  if (is_double_grad &&
      (tmp_var_name.rfind(new_out_suffix) != std::string::npos)) {
    tmp_var_name = tmp_var_name.substr(
        0, tmp_var_name.size() - /*kDoubleGradNewOutSuffix length*/ 4);
  }
  return tmp_var_name.substr(0, tmp_var_name.size() - kGradVarSuffixSize);
}

inline static bool IsGradVar(const std::string& var_name, bool is_double_grad) {
  std::string suffix = kGradVarSuffix;
  if (!is_double_grad) {
    return var_name.rfind(suffix) != std::string::npos;
  } else {
    // for double grad cases, the X@GRAD is not a grad var, X@GRAD@GRAD is a
    // grad var, here we remove a @GRAD suffix
    return NoGrad(var_name).rfind(suffix) != std::string::npos;
  }
}

inline static bool IsMemberOf(const std::vector<std::string>& vec,
                              const std::string& name) {
  return std::find(vec.cbegin(), vec.cend(), name) != vec.cend();
}

}  // namespace detail
}  // namespace framework
}  // namespace paddle
