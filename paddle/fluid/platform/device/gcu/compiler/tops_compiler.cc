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

#include "paddle/fluid/platform/device/gcu/compiler/tops_compiler.h"
#include <string>
#include <vector>

#include "gcu/dtu_compiler/tops_graph_compiler.h"
#include "gcu/dtu_compiler/tops_graph_compiler_option.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"

namespace paddle {
namespace platform {
namespace gcu {

std::vector<std::string> target_option_split(const std::string& s,
                                             char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    std::size_t firstNonSpace = token.find_first_not_of(" \t\n\r");
    std::size_t lastNonSpace = token.find_last_not_of(" \t\n\r");
    if (firstNonSpace == std::string::npos ||
        lastNonSpace == std::string::npos) {
      continue;
    }
    token.substr(firstNonSpace, lastNonSpace - firstNonSpace + 1);
    if (!token.empty()) tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> GetTopsCompileOptions() {
  std::vector<std::string> opts;

  auto target_name = runtime::Context::GlobalTargetName();
  std::string hlir_options =
      "hlir-training-pipeline{tensor-split=true op-key=pavo "
      "dynamic-shape=false}";

  // add target options
  int options_len = 1024;            // NOLINT
  char target_options[options_len];  // NOLINT
  CHECK_EQ(
      TOPS_GRAPH_SUCCESS,
      topsgraphInitOptions(target_name.c_str(), target_options, options_len));
  std::string target_opt_s = std::string(target_options);
  char delimiter = '-';
  auto target_opt_vec = target_option_split(target_opt_s, delimiter);
  for (auto it : target_opt_vec) {
    auto temp_opt = "-" + it;
    opts.push_back(temp_opt);
  }
  opts.push_back(std::string("-hlir=") + hlir_options);

  if (VLOG_IS_ON(6)) {
    std::stringstream ss;
    ss << "compile options: ";
    for (auto it : opts) {
      ss << it << " ";
    }
    VLOG(6) << ss.str();
  }

  return opts;
}

topsExecutable_t CompileExecutable(std::shared_ptr<hlir::Module> module) {
  std::vector<const char*> options;

  auto compile_options = GetTopsCompileOptions();
  for (auto& option : compile_options) {
    options.push_back(option.c_str());
  }

  // create program and compile
  topsgraphProgram program;
  CHECK_EQ(TOPS_GRAPH_SUCCESS,
           topsgraphCreateProgramFromModule(&program, module.get()));

  CHECK_EQ(TOPS_GRAPH_SUCCESS,
           topsgraphCompileProgram(program, options.size(), options.data()));

  // get binary size and binary data
  uint64_t binary_size = 0;
  CHECK_EQ(TOPS_GRAPH_SUCCESS, topsgraphGetBinSize(program, &binary_size));
  std::unique_ptr<char[]> binary(new char[binary_size]);
  CHECK_EQ(TOPS_GRAPH_SUCCESS, topsgraphGetBin(program, binary.get()));

  // delete program
  topsgraphDestroyProgram(&program);

  topsExecutable_t exe;
  RT_CHECK(topsCreateExecutable(&exe, binary.get(), binary_size));

  return exe;
}

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
