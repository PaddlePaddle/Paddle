// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <iostream>
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

static std::string RefineName(std::string name) {
  for (auto &e : name) {
    if (e == '-' || e == '@') {
      e = '_';
    }
  }
  return name;
}

static std::vector<std::string> GenerateOpFunctions(
    const std::string &module_name) {
  auto &op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_function_list;
  for (auto &pair : op_info_map) {
    auto &op_info = pair.second;
    auto op_proto = op_info.proto_;
    if (op_proto == nullptr) {
      continue;
    }

    std::string result =
        "  " + module_name + ".def(\"" + op_proto->type() + "\", [](";
    for (auto &input : op_proto->inputs()) {
      result += ("const paddle::framework::Variable *" +
                 RefineName(input.name()) + ",");
    }

    for (auto &output : op_proto->outputs()) {
      result +=
          ("paddle::framework::Variable *" + RefineName(output.name()) + ",");
    }

    if (result.back() == ',') {
      result.pop_back();
    }

    result += ") {});\n";
    op_function_list.emplace_back(std::move(result));
  }

  return op_function_list;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "argc must be 2" << std::endl;
    return -1;
  }

  std::vector<std::string> headers{};

  std::ofstream out(argv[1], std::ios::out);

  out << "#pragma once\n\n";

  for (auto &header : headers) {
    out << "#include \"" + header + "\"\n";
  }

  out << "namespace paddle {\n"
      << "namespace pybind {\n"
      << "\n"
      << "inline void BindOpFunctions(pybind11::module *module) {\n"
      << "  auto m = module->def_submodule(\"ops\");\n\n";

  auto op_funcs = GenerateOpFunctions("m");
  out << paddle::string::join_strings(op_funcs, '\n');

  out << "}\n\n"
      << "} // namespace pybind\n"
      << "} // namespace paddle\n";

  out.close();
  return 0;
}
