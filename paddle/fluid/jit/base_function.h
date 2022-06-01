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

class Argument {
 public:
  explicit Argument(const std::string &name, bool is_out = false)
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

// TODO(dev): make it as abstract class
class BaseFunction {
 public:
  BaseFunction(const framework::ProgramDesc &prog,
               const std::vector<IValue> &params)
      : prog_(prog), params_(params) {
    // Construct executor.
    Init();
  }
  virtual ~BaseFunction() {}

  virtual std::vector<IValue> operator()(const std::vector<IValue> &args) = 0;

 protected:
  void Init() {
    // Parse FunctionSchema
    skip_vars_ = prog_.GetFetchTargetNames();
    for (auto &in_name : prog_.GetFeedTargetNames()) {
      schema_.input_args.emplace_back(in_name, false);
    }

    for (auto &out_name : skip_vars_) {
      schema_.output_args.emplace_back(out_name, true);
    }
    // share params into scope
    ShareIntoScope(params_);
    VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
    // remove feed fetch op
    RemoveFeedFetch();
  }

  void FetchOutput(std::vector<IValue> *outs) {
    for (auto &out_name : skip_vars_) {
      VLOG(3) << "fetch out: " << out_name;
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
    VLOG(3) << "ivals size: " << ivals.size();
    for (size_t i = 0; i < ivals.size(); ++i) {
      auto &tensor = ivals[i].AsTensor();
      VLOG(3) << "share into scope: " << tensor.name();
      auto *var = scope_.Var(tensor.name());
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
      *dst_tensor = *t;
    }
  }

  void RemoveFeedFetch() {
    for (size_t i = 0; i < prog_.Size(); ++i) {
      auto *block = prog_.MutableBlock(i);
      const auto &all_ops = block->AllOps();
      size_t op_size = all_ops.size();
      VLOG(3) << "op_size: " << op_size;
      for (int i = op_size - 1; i >= 0; i--) {
        auto op = all_ops[i];
        VLOG(3) << "i: " << i << " " << op->Type();
        if (op->Type() == "feed" || op->Type() == "fetch") {
          VLOG(3) << "remove op type: " << op->Type() << ", index: " << i;
          block->RemoveOp(i, i + 1);
        }
      }
    }
  }

 protected:
  framework::ProgramDesc prog_;
  // TODO(dev): need a better way to share params
  const std::vector<IValue> &params_;
  std::vector<std::string> skip_vars_;
  FunctionSchema schema_;
  // global_scope place params
  framework::Scope scope_;
  //   framework::Executor inner_exe_;
};

}  // namespace jit
}  // namespace paddle
