// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/extern_func_protos.h"

#include <string>
#include <vector>

namespace cinn {
namespace backends {

ExternFunctionProtoRegistry::ExternFunctionProtoRegistry() {
  static const std::vector<std::string> extern_funcs_fp32_unary = {
      "exp",  "erf",   "sigmoid", "sqrt", "log",   "log2", "log10", "floor",
      "ceil", "round", "trunc",   "cos",  "cosh",  "tan",  "tanh",  "sin",
      "sinh", "acos",  "acosh",   "asin", "asinh", "atan", "atanh", "fabs"};
  static const std::vector<std::string> extern_funcs_float_bool_unary = {
      "isnan", "isfinite", "isinf"};
  static const std::vector<std::string> extern_funcs_int_binary = {
      "left_shift",
      "right_shift",
      "bitwise_or",
      "bitwise_and",
      "bitwise_xor",
      "bitwise_not"};
  static const std::vector<std::string> extern_funcs_int_int_unary = {
      "bitwise_not"};
  for (int i = 0; i < extern_funcs_fp32_unary.size(); ++i) {
    auto* proto =
        new FunctionProto(extern_funcs_fp32_unary[i], {Float(32)}, Float(32));
    Register(proto->name, proto);
  }
  for (int i = 0; i < extern_funcs_float_bool_unary.size(); ++i) {
    auto* proto = new FunctionProto(
        extern_funcs_float_bool_unary[i], {Float(32)}, Bool());
    Register(proto->name, proto);
  }
  for (int i = 0; i < extern_funcs_int_binary.size(); ++i) {
    auto* proto = new FunctionProto(
        extern_funcs_int_binary[i], {Int(32), Int(32)}, Int(32));
    Register(proto->name, proto);
  }
  for (int i = 0; i < extern_funcs_int_int_unary.size(); ++i) {
    auto* proto =
        new FunctionProto(extern_funcs_int_int_unary[i], {Int(32)}, Int(32));
    Register(proto->name, proto);
  }

  auto* n = detail::CreateTanhVProto();
  Register(n->name, n);
}

ExternFunctionProtoRegistry& ExternFunctionProtoRegistry::Global() {
  static ExternFunctionProtoRegistry x;
  return x;
}

namespace detail {

FunctionProto* CreateTanhVProto() {
  return new FunctionProto(extern_func__tanh_v,
                           {type_of<float*>()},
                           {type_of<float*>()},
                           Void(),
                           FunctionProto::ShapeFollowNthArgument(0));
}

}  // namespace detail
}  // namespace backends
}  // namespace cinn
