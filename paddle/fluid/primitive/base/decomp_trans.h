// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/pir/dialect/operator/interface/decomp.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/program.h"

namespace paddle {

class DecompProgram {
 public:
  explicit DecompProgram(pir::Program* program) : program_(program) {}

  DecompProgram(pir::Program* program,
                const std::vector<pir::Value>& src_vars,
                const std::set<std::string>& blacklist,
                const std::set<std::string>& whitelist)
      : program_(program),
        src_vars_(src_vars),
        blacklist_(blacklist),
        whitelist_(whitelist) {}

  void decomp_program();
  bool check_decomp_dynamic_shape(pir::Operation* op);
  void check_decomp_outputs(const std::string& op_name,
                            const std::vector<pir::Value>& orig_outs,
                            const std::vector<pir::Value>& decomp_outs);
  std::vector<pir::Value> format_decomp_res(
      const std::string& op_name,
      const std::vector<pir::Value>& orig_outs,
      const std::vector<std::vector<pir::Value>>& decomp_outs);
  std::vector<pir::Value> construct_dst_vars(
      const std::string& op_name,
      const std::vector<pir::Value>& orig_outs,
      const std::vector<pir::Value>& decomp_outs,
      std::unordered_map<pir::Value, int> orig_vars_dict);
  bool enable_decomp_by_filter(const std::string& op_name);
  void set_src_vars(const std::vector<pir::Value>& src_vars) {
    src_vars_ = src_vars;
  }
  void set_blacklist(const std::set<std::string>& blacklist) {
    blacklist_ = blacklist;
  }
  void set_whitelist(const std::set<std::string>& whitelist) {
    whitelist_ = whitelist;
  }
  std::vector<pir::Value> get_dst_vars();

 private:
  pir::Program* program_;
  std::vector<pir::Value> src_vars_;
  std::vector<pir::Value> dst_vars_;
  std::set<std::string> blacklist_;
  std::set<std::string> whitelist_;
};

bool has_decomp_rule(const pir::Operation& op);

std::vector<std::vector<pir::Value>> call_decomp_rule(pir::Operation* op);

}  // namespace paddle
