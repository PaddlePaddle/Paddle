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
#include <map>
#include <vector>
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace pybind {

/**
 * A base context that represents the CINN IR that need context information
 */
class IRContextNode : public cinn::common::Object {
 public:
  std::vector<ir::Expr> exprs;

 public:
  // Corresponds to the __enter__ method in python's context manager
  virtual void EnterWithContext();
  // Corresponds to the __exit__ method in python's context manager
  virtual void ExitWithContext();
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr char* __type_info__ = "IRContextNode";
};

/**
 * The life cycle of RAII resource management for IRContextNode
 * is determined at the Python.
 */
class IRContext {
 public:
  IRContext() = default;
  IRContext(const IRContext& other) = default;
  explicit IRContext(IRContextNode* x) : data_(x) {}

  const IRContextNode* get() const { return data_.get(); }
  const IRContextNode* operator->() const { return data_.get(); }

  void add_expr(Expr expr) { data_->exprs.push_back(expr); }

 public:
  cinn::common::Shared<IRContextNode> data_;

 public:
  template <typename TIRContextNode>
  const TIRContextNode* As() const {
    static_assert(std::is_base_of<IRContextNode, TIRContextNode>());
    PADDLE_ENFORCE_NOT_NULL(
        data_.get(), ::common::errors::InvalidArgument("IrContext holds null"));
    auto* ctx_node = data_.get()->safe_as<TIRContextNode>();
    if (!ctx_node) {
      std::stringstream err_msg;
      err_msg << "TypeConvertError: convert " << data_.get()->type_info()
              << " to " << TIRContextNode::__type_info__;

      PADDLE_THROW(::common::errors::InvalidArgument(err_msg.str()));
    }
    return ctx_node;
  }
  template <typename TIRContextNode>
  TIRContextNode* As() {
    PADDLE_ENFORCE_NOT_NULL(
        data_.get(), ::common::errors::InvalidArgument("IrContext holds null"));
    auto* ctx_node = data_.get()->safe_as<TIRContextNode>();
    if (!ctx_node) {
      std::stringstream ss;
      ss << "TypeConvertError: convert " << data_.get()->type_info() << " to "
         << TIRContextNode::__type_info__;
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    }
    return ctx_node;
  }
};

class ScheduleBlockContextNode : public IRContextNode {
 public:
  std::vector<Var> iter_vars;
  // BufferRange(s) which is read in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> read_buffers;
  // BufferRange(s) which is written in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> write_buffers;
  // Additional attributes about this schedulable block,
  // which take some auxiliary hints for future transformations.
  std::map<std::string, ir::attr_t> attrs;
  // values of the iter_vars
  std::vector<Expr> iter_values;
  std::string name;

 public:
  ScheduleBlockContextNode() = default;
  explicit ScheduleBlockContextNode(std::string name) : name(name) {}
  void ExitWithContext() final;
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr const char* __type_info__ = "ScheduleBlockContextNode";
};

class ScheduleBlockContext : public IRContext {
 public:
  explicit ScheduleBlockContext(ScheduleBlockContextNode* x) : IRContext(x) {}
};

class ForContextNode : public IRContextNode {
 public:
  //! The loop variable.
  Var loop_var;
  //! The minimum value of the iteration.
  Expr min;
  //! The extent of the iteration.
  Expr extent;

 public:
  void ExitWithContext() final;
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr const char* __type_info__ = "ForContextNode";
};

class LowerFuncContextNode : public IRContextNode {
 public:
  //! The name of this function.
  std::string name;
  //! The Arguments used in the body of the function.
  std::vector<ir::Argument> args;

 public:
  LowerFuncContextNode() = default;
  explicit LowerFuncContextNode(std::string name) : name(name) {}
  void ExitWithContext() final;
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr const char* __type_info__ = "LowerFuncContextNode";
};

class IfContextNode : public IRContextNode {
 public:
  Expr condition;
  Expr true_case;
  Expr false_case;

 public:
  IfContextNode() = default;
  explicit IfContextNode(Expr condition)
      : condition(condition), true_case(Expr()), false_case(Expr()) {}
  const char* type_info() const override { return __type_info__; }

  void ExitWithContext() final;

 public:
  static constexpr const char* __type_info__ = "IfContextNode";
};

class ThenContextNode : public IRContextNode {
 public:
  ThenContextNode() = default;
  const char* type_info() const override { return __type_info__; }

  void ExitWithContext() final;

 public:
  static constexpr const char* __type_info__ = "ThenContextNode";
};

class ElseContextNode : public IRContextNode {
 public:
  ElseContextNode() = default;
  const char* type_info() const override { return __type_info__; }
  void ExitWithContext() final;

 public:
  static constexpr const char* __type_info__ = "ElseContextNode";
};

/**
 * A stack used to store current IRContext
 */
class IRBuilderNode : public cinn::common::Object {
 public:
  std::vector<IRContext> contexts;
  ir::LoweredFunc result;
  const char* type_info() const override { return __type_info__; }
  ir::LoweredFunc GetResult() const;
  void Reset();

  template <typename TIRContextNode>
  IRContext GetLastContext() const;

  template <typename TIRContextNode>
  IRContext FindContext() const;

 public:
  static constexpr const char* __type_info__ = "IRBuilderNode";
};

/**
 * The life cycle of RAII resource management for IRBuilderNode
 * is determined at the Python.
 */
class IRBuilder {
 public:
  IRBuilder();
  void EnterWithContext();
  void ExitWithContext();
  static IRBuilder CurrentIRBuilder();

 public:
  cinn::common::Shared<IRBuilderNode> data_;
};

std::vector<IRBuilder>* IRBuilderStack();
void LinkToParentContext(ir::Expr);

template <typename TIRContextNode>
IRContext IRBuilderNode::GetLastContext() const {
  if (!(contexts.back().As<TIRContextNode>())) {
    std::stringstream ss;
    ss << "TypeError: The last context is not "
       << TIRContextNode::__type_info__;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return contexts.back();
}

template <typename TIRContextNode>
IRContext IRBuilderNode::FindContext() const {
  for (auto it = contexts.rbegin(); it != contexts.rend(); ++it) {
    if (const TIRContextNode* p = it->As<TIRContextNode>()) {
      return *it;
    }
  }
  return IRContext();
}

}  // namespace pybind

}  // namespace cinn
