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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

std::map<std::string, std::set<std::string>> op_ins_map = {
    {"layer_norm", {"X", "Scale", "Bias"}},
    {"gru_unit", {"Input", "HiddenPrev", "Weight", "Bias"}},
    {"label_smooth", {"X", "PriorDist"}},
    {"assign", {"X"}},
};
std::map<std::string, std::set<std::string>> op_passing_out_map = {
    {"sgd", {"ParamOut"}},
    {"adam", {"ParamOut"}},
    {"momentum", {"ParamOut", "VelocityOut"}},
    {"batch_norm", {"MeanOut", "VarianceOut"}}};
// clang-format off
const char* OUT_INITIALIZER_TEMPLATE =
    R"({"%s", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase(tracer->GenerateUniqueName()))}})";

const char* OUT_DUPLICABLE_INITIALIZER_TEMPLATE =
     R"({"%s", ConstructDuplicableOutput(%s)})";


const char* INPUT_INITIALIZER_TEMPLATE =
    R"({"%s", {%s}})";

// if inputs is list, no need {}
const char* INPUT_LIST_INITIALIZER_TEMPLATE =
    R"({"%s", %s})";

// if inputs is list, no need {}
const char* ARG_OUT_NUM =
    R"(%sNum)";
const char* ARG_OUT_NUM_TYPE =
    R"(size_t )";

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

const char* VAR_TYPE = R"(std::shared_ptr<imperative::VarBase>)";
const char* VAR_LIST_TYPE = R"(std::vector<std::shared_ptr<imperative::VarBase>>)";

const char* ATTR_TYPE = R"(framework::Attribute)";
const char* ARG_TEMPLATE = R"(%s %s)";

const char* RETURN_TUPLE_TYPE = R"(std::tuple<%s>)";
const char* RETURN_TYPE = R"(%s)";

const char* RETURN_TUPLE_TEMPLATE = R"(std::make_tuple(%s))";
const char* RETURN_LIST_TEMPLATE = R"(outs_["%s"])";
const char* RETURN_TEMPLATE = R"(outs_["%s"][0])";

// like elementwise_*, no list in args and only one result in return.
const char* OP_FUNCTION_NO_LIST_SINGLE_RETURN_TEMPLATE =
R"(
inline std::shared_ptr<imperative::VarBase> %s(%s, const framework::AttributeMap& attrs, 
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


const char* FUNCTION_ARGS = R"(%s, const py::args& args)";
const char* FUNCTION_ARGS_NO_INPUT = R"(const py::args& args)";

// like elementwise_*, no list in args and only one result in return.
// return_type, func_name, inputs_args, outs_initializer, ins_initializer,
//   op_name, return_str
const char* OP_FUNCTION_TEMPLATE2 =
R"(
%s %s(%s)
{
  framework::AttributeMap attrs_;
  ConstructAttrMapFromPyArgs(&attrs_, args);
  {
    py::gil_scoped_release release;
    
    auto tracer = imperative::GetCurrentTracer();
    imperative::NameVarBaseMap outs_ = %s;
    imperative::NameVarBaseMap ins_ = %s;

    tracer->TraceOp("%s", ins_, outs_, attrs_);
    return %s; 
  }   
})";

const char* PYBIND_ITEM_TEMPLATE =
R"(
  %s.def("%s", &%s);)";
const char* PYBIND_PY_ARG_TEMPLATE = R"(py::arg("%s"))";
// clang-format on

const std::vector<std::string> specialization = {
    "concat", "elementwise_add", "elementwise_div", "elementwise_max",
    "elementwise_mul",
    //  "fill_constant",
    //  "lookup_table",
    "matmul", "reduce_mean", "reduce_sum", "reshape2", "sgd", "sigmoid",
    "slice", "softmax_with_cross_entropy", "split", "square", "sqrt", "tanh",
    "transpose2",
    //  "uniform_random"
    "conv2d", "cross_entropy2", "mean", "pool2d", "relu", "softmax",
    // "batch_norm", "top_k", "accuracy", "gaussian_random"
};

static bool FindInputInSpecialization(const std::string op_type,
                                      const std::string in_name) {
  return op_ins_map[op_type].count(in_name);
}

static bool FindOutoutInSpecialization(const std::string op_type,
                                       const std::string out_name) {
  return op_passing_out_map[op_type].count(out_name);
}

static std::tuple<std::vector<std::string>, std::vector<std::string>>
GenerateOpFunctions2(const std::string& module_name) {
  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_function_list, bind_function_list;
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();

  for (auto& pair : op_info_map) {
    auto& op_info = pair.second;
    auto op_proto = op_info.proto_;
    if (op_proto == nullptr) {
      continue;
    }

    auto& op_type = op_proto->type();

    // Skip ooerator which is not inherit form OperatorWithKernel, like while,
    // since only OperatorWithKernel can run in dygraph mode.
    if (!all_kernels.count(op_type)) {
      continue;
    }

    std::string input_args = "";
    std::string ins_initializer = "{";
    std::string py_arg = "";
    for (auto& input : op_proto->inputs()) {
      auto& in_name = input.name();
      // skip those dispensable inputs, like ResidualData in conv2d
      if (input.dispensable() && !FindInputInSpecialization(op_type, in_name)) {
        continue;
      }
      // If input is duplicable, use list
      auto in_type = input.duplicable() ? VAR_LIST_TYPE : VAR_TYPE;

      auto input_arg = paddle::string::Sprintf(ARG_TEMPLATE, in_type, in_name);
      input_args += input_arg;
      input_args += ",";

      auto in_template = input.duplicable() ? INPUT_LIST_INITIALIZER_TEMPLATE
                                            : INPUT_INITIALIZER_TEMPLATE;
      ins_initializer += paddle::string::Sprintf(in_template, in_name, in_name);
      ins_initializer += ",";
      // py_arg += paddle::string::Sprintf(PYBIND_PY_ARG_TEMPLATE, in_name);
      // py_arg += ",";
    }

    if (ins_initializer.back() == ',') {
      ins_initializer.pop_back();
    }
    ins_initializer += "}";

    if (input_args.back() == ',') {
      input_args.pop_back();
      // py_arg.pop_back();
    }

    // Generate outs initializer
    std::string outs_initializer = "{";
    std::string return_type = "";
    std::string return_str = "";

    int outs_num = 0;
    for (auto& output : op_proto->outputs()) {
      if (output.dispensable()) {
        continue;
      }

      auto out_type = output.duplicable() ? VAR_LIST_TYPE : VAR_TYPE;
      auto return_template =
          output.duplicable() ? RETURN_LIST_TEMPLATE : RETURN_TEMPLATE;

      auto& out_name = output.name();
      std::string out_initializer_str;
      if (FindOutoutInSpecialization(op_type, out_name)) {
        input_args += ",";
        input_args += out_type;
        input_args += out_name;
        auto out_template = output.duplicable()
                                ? INPUT_LIST_INITIALIZER_TEMPLATE
                                : INPUT_INITIALIZER_TEMPLATE;
        out_initializer_str +=
            paddle::string::Sprintf(out_template, out_name, out_name);
      } else {
        if (output.duplicable()) {
          if (input_args != "") {
            input_args += ",";
          }
          auto out_num_str = paddle::string::Sprintf(ARG_OUT_NUM, out_name);
          input_args += ARG_OUT_NUM_TYPE;
          input_args += out_num_str;
          out_initializer_str = paddle::string::Sprintf(
              OUT_DUPLICABLE_INITIALIZER_TEMPLATE, out_name, out_num_str);
        } else {
          out_initializer_str =
              paddle::string::Sprintf(OUT_INITIALIZER_TEMPLATE, out_name);
        }
      }

      return_type += out_type;
      return_type += ",";
      return_str += paddle::string::Sprintf(return_template, out_name);
      return_str += ",";
      outs_num += 1;
      // There are few Operators that have duplicable output, like `Out` in
      // split op. We need to specify the number of variables for the duplicable
      // output, as the argument OutNum;

      outs_initializer += out_initializer_str;
      outs_initializer += ",";
    }
    if (outs_initializer.back() == ',') {
      outs_initializer.pop_back();
      return_type.pop_back();
      return_str.pop_back();
    }
    outs_initializer += "}";
    if (outs_num == 0) {
      return_type = "void";
    }
    if (outs_num > 1) {
      return_str = paddle::string::Sprintf(RETURN_TUPLE_TEMPLATE, return_str);
      return_type = paddle::string::Sprintf(RETURN_TUPLE_TYPE, return_type);
    }
    std::string function_args = "";
    if (input_args == "") {
      function_args =
          paddle::string::Sprintf(FUNCTION_ARGS_NO_INPUT, input_args);
    } else {
      function_args = paddle::string::Sprintf(FUNCTION_ARGS, input_args);
    }

    std::string func_name = "imperative_" + op_type;
    // generate op funtcion body
    // return_type, func_name, inputs_args, outs_initializer, ins_initializer,
    // op_name, return_str
    auto op_function_str = paddle::string::Sprintf(
        OP_FUNCTION_TEMPLATE2, return_type, func_name, function_args,
        outs_initializer, ins_initializer, op_type, return_str);
    // std::cout << op_function_str << std::endl;

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

  auto op_funcs = GenerateOpFunctions2("m");
  // all op functions
  // auto op_funcs = GenerateOpFunctions("m");

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
