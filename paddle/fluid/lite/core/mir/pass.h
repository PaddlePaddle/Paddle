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
#include "paddle/fluid/lite/core/mir/node.h"
#include "paddle/fluid/lite/core/mir/ssa_graph.h"

namespace paddle {
namespace lite {
namespace mir {

class Pass {
 public:
  // Some appoint here, one pass should be only one of the following kinds.
  enum class Kind {
    // Will modify the program/graph topology.
    kProgramWise = 0,
    // Will modify the instruction, with the graph topology fixed.
    kInstructionWise,
    // Will not modify the IR, just collect information or visualization.
    kDebug,
  };

  Pass(Kind kind) : kind_(kind) {}

  virtual void Apply(std::unique_ptr<mir::SSAGraph>& graph) = 0;

  void set_name(const std::string& name) { name_ = name; }
  const std::string& name() const { return name_; }

  void set_doc(const std::string& doc) { doc_ = doc; }
  const std::string& doc() const { return doc_; }

  Kind kind() const { return kind_; }
  bool is_debug_pass() const { return kind_ == Kind::kDebug; }
  bool is_program_pass() const { return kind_ == Kind::kProgramWise; }
  bool is_instruction_pass() const { return kind_ == Kind::kInstructionWise; }

  virtual ~Pass() = default;

 private:
  const Kind kind_;
  std::string name_;
  std::string doc_;
};

// Different kinds.
class ProgramPass : public Pass {
 public:
  ProgramPass() : Pass(Kind::kProgramWise) {}
};

class InstructionPass : public Pass {
 public:
  InstructionPass() : Pass(Kind::kInstructionWise) {}
};

class DebugPass : public Pass {
 public:
  DebugPass() : Pass(Kind::kDebug) {}
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
