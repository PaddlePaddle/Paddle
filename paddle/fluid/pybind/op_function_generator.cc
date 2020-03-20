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
#include <string>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

// clang-format off
const char* OUT_INITIALIZER_TEMPLATE =
    R"({"%s", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase(tracer->GenerateUniqueName()))}})";

const char* OP_FUNCTION_TEMPLATE =
R"(
inline imperative::NameVarBaseMap %s(const imperative::NameVarBaseMap& ins, const framework::AttributeMap& attrs, 
  imperative::NameVarBaseMap outs, const std::map<std::string, size_t>& out_nums)
{
  auto tracer = imperative::GetCurrentTracer();
  if (outs.size() == 0) {
    if (out_nums.size() == 0) {
      imperative::NameVarBaseMap outs_ = %s;
      outs = std::move(outs_);
    } else {
      for (auto &pair : out_nums) {
        for (size_t i = 0; i < pair.second; i ++) {
          auto var_base_name = tracer->GenerateUniqueName();
          outs[pair.first].emplace_back(new imperative::VarBase(var_base_name));
        }
      }
    }
  }
  
  tracer->TraceOp("%s", std::move(ins), std::move(outs), std::move(attrs));
  return outs;
})";

const char* PYBIND_ITEM_TEMPLATE =
R"(
  %s.def("%s", &%s, py::arg("ins"), py::arg("attrs")=framework::AttributeMap(), py::arg("outs")=imperative::NameVarBaseMap(), 
    py::arg("out_nums")=std::map<std::string, size_t>(), py::call_guard<py::gil_scoped_release>());)";

// clang-format on

static std::tuple<std::vector<std::string>, std::vector<std::string>>
GenerateOpFunctions(const std::string& module_name) {
  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_function_list, bind_function_list;
  for (auto& pair : op_info_map) {
    auto& op_info = pair.second;
    auto op_proto = op_info.proto_;
    if (op_proto == nullptr) {
      continue;
    }
    auto& op_type = op_proto->type();

    // Generate outs initializer
    std::string outs_initializer = "{";

    for (auto& output : op_proto->outputs()) {
      auto& out_name = output.name();
      auto out_initializer_str =
          paddle::string::Sprintf(OUT_INITIALIZER_TEMPLATE, out_name);
      outs_initializer += out_initializer_str;
      outs_initializer += ",";
    }
    if (outs_initializer.back() == ',') {
      outs_initializer.pop_back();
    }
    outs_initializer += "}";

    std::string func_name = "imperative_" + op_type;

    // generate op funtcion body
    auto op_function_str = paddle::string::Sprintf(
        OP_FUNCTION_TEMPLATE, func_name, outs_initializer, op_type);

    // generate pybind item
    auto bind_function_str = paddle::string::Sprintf(
        PYBIND_ITEM_TEMPLATE, module_name, op_type, func_name);

    op_function_list.emplace_back(std::move(op_function_str));
    bind_function_list.emplace_back(std::move(bind_function_str));
  }

  return std::make_tuple(op_function_list, bind_function_list);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "argc must be 2" << std::endl;
    return -1;
  }

  std::vector<std::string> headers{"\"paddle/fluid/imperative/tracer.h\""};

  std::ofstream out(argv[1], std::ios::out);

  out << "#pragma once\n\n";

  for (auto& header : headers) {
    out << "#include  " + header + "\n";
  }

  // all op functions
  auto op_funcs = GenerateOpFunctions("m");

  out << "namespace py = pybind11;"
      << "\n";
  out << "namespace paddle {\n"
      << "namespace pybind {\n";
  out << paddle::string::join_strings(std::get<0>(op_funcs), '\n');
  out << "\n\n";

  out << "inline void BindOpFunctions(pybind11::module *module) {\n"
      << "  auto m = module->def_submodule(\"ops\");\n\n";

  out << paddle::string::join_strings(std::get<1>(op_funcs), '\n');
  out << "\n";
  out << "}\n\n"
      << "} // namespace pybind\n"
      << "} // namespace paddle\n";

  out.close();
  return 0;
}
