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

const int BUFFER_SIZE = 4096;

const char* INPUT_TYPE = R"(imperative::NameVarBaseMap)";
const char* ARGS_TYPE = R"(framework::AttributeMap)";

const char* OUT_VAR_NUM_TYPE = R"(std::map<std::string, size_t>)";
const char* RETURN_TYPE = R"(imperative::NameVarBaseMap)";

const char* OUTS_INITIALIZER =
    R"({{"out": VarBase("mul_out")}, "XX": VarBase("mul_xx")}})";

const char* OUT_INITIALIZER_TEMPLATE =
    R"({"%s", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase(tracer->GenerateUniqueName())) } })";

const char* OP_FUNCTION_TEMPLATE =
    R"([](const imperative::NameVarBaseMap& ins, const framework::AttributeMap& attrs, 
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
            auto out = new imperative::VarBase(var_base_name);
            outs[pair.first].emplace_back(std::shared_ptr<imperative::VarBase>(out));
          }
        }
      }
    }
    
    {
      py::gil_scoped_release release;
      tracer->TraceOp("%s", std::move(ins), std::move(outs), std::move(attrs));
      return outs;
    }
  }, py::arg("ins"), py::arg("attrs")=framework::AttributeMap(), 
  py::arg("outs")=imperative::NameVarBaseMap(), 
  py::arg("out_nums")=std::map<std::string, size_t>())";

const char* PYBIND_TEMPLATE = R"(%s.def("%s", %s);)";

static std::string RefineName(std::string name) {
  for (auto& e : name) {
    if (e == '-' || e == '@') {
      e = '_';
    }
  }
  return name;
}

static std::vector<std::string> GenerateOpFunctions(
    const std::string& module_name) {
  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_function_list;
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
      char out_initializer_buf[BUFFER_SIZE];
      snprintf(out_initializer_buf, BUFFER_SIZE, OUT_INITIALIZER_TEMPLATE,
               out_name.c_str(), op_type.c_str(), out_name.c_str());
      outs_initializer += out_initializer_buf;
      outs_initializer += ",";
    }
    if (outs_initializer.back() == ',') {
      outs_initializer.pop_back();
    }
    outs_initializer += "}";

    // generate op funtcion body
    char op_function_buf[BUFFER_SIZE];
    snprintf(op_function_buf, BUFFER_SIZE, OP_FUNCTION_TEMPLATE,
             outs_initializer.c_str(), op_type.c_str());

    // generate pybind line
    char pybind_buf[BUFFER_SIZE];
    snprintf(pybind_buf, BUFFER_SIZE, PYBIND_TEMPLATE, module_name.c_str(),
             op_type.c_str(), op_function_buf);

    std::string pybind_op_function = pybind_buf;
    pybind_op_function += "\n";

    op_function_list.emplace_back(std::move(pybind_op_function));
  }

  return op_function_list;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "argc must be 2" << std::endl;
    return -1;
  }

  std::vector<std::string> headers{};

  std::ofstream out(argv[1], std::ios::out);

  out << "#pragma once\n\n";

  for (auto& header : headers) {
    out << "#include  " + header + "\n";
  }
  // out << UNIQUE_NAME_GENERATOR << "\n";
  out << "namespace py = pybind11;"
      << "\n";
  out << "namespace paddle {\n"
      << "namespace pybind {\n"
      << "\n"
      << "inline void BindOpFunctions(pybind11::module *module) {\n"
      << "  auto m = module->def_submodule(\"ops\");\n\n";

  // all op functions
  auto op_funcs = GenerateOpFunctions("m");

  out << paddle::string::join_strings(op_funcs, '\n');

  out << "}\n\n"
      << "} // namespace pybind\n"
      << "} // namespace paddle\n";

  out.close();
  return 0;
}
