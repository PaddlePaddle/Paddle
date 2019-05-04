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

#pragma once
#include <list>
#include <string>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/mir/node.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

static const std::string kKernelTypeAttr = "__@kernel_type_attr@__";

// A program is used to represent a code program, in Paddle, a code program
// contains:
// - main block, which is a list of OpLite
// - scope: which contains all the weights
struct Program {
  std::list<std::string> tmp_vars;
  std::list<std::string> weights;
  std::list<std::shared_ptr<OpLite>> ops;
  // the scope to run the kernels, NOTE this is the execution scope.
  std::shared_ptr<lite::Scope> scope;
  std::vector<Place> valid_places;
  // Runtime scope.
  lite::Scope* exec_scope{};
  const framework::proto::ProgramDesc desc;

  explicit Program(const std::shared_ptr<Scope>& root) { scope = root; }
  Program(const framework::proto::ProgramDesc& desc,
          const std::shared_ptr<Scope>& root,
          const std::vector<Place>& valid_places)
      : scope(root), valid_places(valid_places), desc(desc) {
    CHECK(scope) << "scope should be init first";
    PrepareWorkspace(desc);
    Build(desc);
  }

  std::unique_ptr<Program> Clone() const {
    std::unique_ptr<Program> res(new Program(desc, scope, valid_places));
    return res;
  }

 private:
  // Build from a program and scope.
  void Build(const framework::proto::ProgramDesc& program) {
    CHECK(ops.empty()) << "Executor duplicate Build found";

    // Create operators.
    for (const auto& proto_op_desc : program.blocks(0).ops()) {
      lite::OpDesc op_desc(proto_op_desc);
      auto op_type = op_desc.Type();
      // if (op_type == "feed" || op_type == "fetch") continue;
      VLOG(4) << "create Op [" << op_type << "]";
      auto op = LiteOpRegistry::Global().Create(op_type);
      CHECK(op) << "no Op found for " << op_type;
      ops.emplace_back(op);
      ops.back()->Attach(op_desc, exec_scope);
    }
  }

  // Create temporary variables.
  void PrepareWorkspace(const framework::proto::ProgramDesc& program) {
    CHECK(!exec_scope) << "Duplicate PrepareWorkspace found";
    exec_scope = &scope->NewScope();
    // Create Feed and Fetch var.
    scope->Var("feed")->GetMutable<std::vector<Tensor>>();
    scope->Var("fetch")->GetMutable<std::vector<Tensor>>();

    tmp_vars.push_back("feed");
    tmp_vars.push_back("fetch");
    for (auto proto_var_desc : program.blocks(0).vars()) {
      lite::VarDesc var_desc(proto_var_desc);
      if (!var_desc.Persistable()) {
        tmp_vars.push_back(var_desc.Name());
        exec_scope->Var(var_desc.Name());
      } else {
        if (var_desc.Name() == "feed" || var_desc.Name() == "fetch") continue;
        weights.push_back(var_desc.Name());
      }
    }
  }
};

struct Instruct {
  Instruct(const std::shared_ptr<OpLite>& op,
           std::unique_ptr<KernelBase>&& kernel)
      : op_(op), kernel_(std::move(kernel)) {}

  void Run() {
    CHECK(op_);
    CHECK(kernel_);
    if (UNLIKELY(first_epoch_)) {
      first_epoch_ = false;
      CHECK(op_->CheckShape());
    }
    op_->InferShape();
    kernel_->Run();
  }

  friend std::ostream& operator<<(std::ostream& os, const Instruct& other) {
    os << other.kernel_->summary() << "\t(" << other.kernel_->doc() << ")";
    return os;
  }

  const OpLite* op() const { return op_.get(); }
  const KernelBase* kernel() const { return kernel_.get(); }

 private:
  std::shared_ptr<OpLite> op_;
  std::unique_ptr<KernelBase> kernel_;
  bool first_epoch_{true};
};

/*
 * A program contains kernels for runtime.
 */
class RuntimeProgram {
 public:
  explicit RuntimeProgram(std::vector<Instruct>&& insts)
      : instructions_(std::move(insts)) {
    if (instructions_.empty()) {
      LOG(FATAL) << "no instructions";
    }
  }

  void Run() {
    for (auto& inst : instructions_) {
      LOG(INFO) << ">> Running kernel: " << inst;
      inst.Run();
    }
  }

  // Serialize the graph and save to the disk.
  void PersistModel(const std::string& dir,
                    const framework::proto::ProgramDesc& desc);

  void set_exec_scope(lite::Scope* x) { exec_scope_ = x; }
  lite::Scope* exec_scope() { return exec_scope_; }

  size_t num_instructions() const { return instructions_.size(); }

 protected:
  std::string SerializeProgram(const framework::proto::ProgramDesc& desc);
  void SaveParams(const std::string& dir,
                  const framework::proto::ProgramDesc& desc);

 private:
  RuntimeProgram(const RuntimeProgram&) = delete;
  std::vector<Instruct> instructions_;
  lite::Scope* exec_scope_{};
};

}  // namespace lite
}  // namespace paddle
