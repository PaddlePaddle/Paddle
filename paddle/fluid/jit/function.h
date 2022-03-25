// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/jit/ivalue.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/none.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace jit {

using std::cout;
using std::endl;

class Argument {
 public:
  Argument(const std::string &name, bool is_out = false)
      : name_(name), is_output_(is_out) {}

  const std::string &Name() const { return name_; }

 private:
  std::string name_;
  // paddle::optional<IValue> default_val_;
  bool is_output_;
};

class FunctionSchema {
 public:
  FunctionSchema() = default;

  std::vector<Argument> input_args;
  std::vector<Argument> output_args;
};

// TODO: make it as abstract class
class Function {
 public:
  Function(const framework::ProgramDesc &prog, std::vector<IValue> &params)
      : prog_(prog), params_(params), inner_exe_(phi::CPUPlace()) {
    // Construct executor.
    Init();
  }
  ~Function() {}

  // TODO: make it virtual
  std::vector<IValue> operator()(const std::vector<IValue> &args) {
    // share input into scope
    ShareIntoScope(args);
    // run program
    inner_exe_.Run(prog_, &scope_, /*blockID=*/0, false, true, skip_vars_);
    // fetch outputs
    std::vector<IValue> res;
    FetchOutput(&res);
    return res;
  }

 private:
  void Init() {
    // Parse FunctionSchema
    skip_vars_ = prog_.GetFetchTargetNames();
    for (auto &in_name : prog_.GetFeedTargetNames()) {
      cout << "feed name: " << in_name << endl;
      schema_.input_args.emplace_back(in_name, false);
    }

    for (auto &out_name : skip_vars_) {
      cout << "fetch name: " << out_name << endl;
      schema_.output_args.emplace_back(out_name, true);
    }
    // share params into scope
    ShareIntoScope(params_);
    cout << framework::GenScopeTreeDebugInfo(&scope_) << endl;

    // remove feed fetch op
    RemoveFeedFetch();
  }

  void FetchOutput(std::vector<IValue> *outs) {
    for (auto &out_name : skip_vars_) {
      cout << "fetch out: " << out_name << endl;
      auto *var = scope_.FindVar(out_name);
      auto &src_tensor = var->Get<phi::DenseTensor>();
      Tensor t(std::make_shared<phi::DenseTensor>());
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(t.impl().get()));
      *dst_tensor = src_tensor;
      outs->emplace_back(t);
    }
  }

  void ShareIntoScope(const std::vector<IValue> &ivals) {
    cout << "ivals size: " << ivals.size() << endl;
    for (size_t i = 0; i < ivals.size(); ++i) {
      auto &tensor = ivals[i].AsTensor();
      cout << "share into scope: " << tensor.name() << endl;
      auto *var = scope_.Var(tensor.name());
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
      *dst_tensor = *t;
    }
  }

  void RemoveFeedFetch() {
    for (size_t i = 0; i < prog_.Size(); ++i) {
      auto *block = prog_.MutableBlock(i);
      size_t idx = 0;
      for (auto *op : block->AllOps()) {
        if (op->Type() == "feed" || op->Type() == "fetch") {
          cout << "remove op: " << idx;
          block->RemoveOp(idx, idx + 1);
        }
        idx++;
      }
    }
  }

 private:
  framework::ProgramDesc prog_;
  // TODO: need a better way to share params
  const std::vector<IValue> &params_;
  std::vector<std::string> skip_vars_;
  FunctionSchema schema_;
  // global_scope place params
  framework::Scope scope_;
  framework::Executor inner_exe_;
};

}  // namespace jit
}  // namespace paddle
