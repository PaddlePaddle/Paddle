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
#include <stack>

#include "paddle/common/macros.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/parameter.h"
#include "paddle/pir/include/core/program.h"

namespace paddle {
namespace dialect {
///
/// \brief ApiBuilder is used in IR API for building op
///
class ApiBuilder {
 public:
  static ApiBuilder& Instance() {
    static ApiBuilder api_builder;
    return api_builder;
  }
  void SetProgram(pir::Program* program);

  void ResetInsertionPointToStart();

  void ResetInsertionPointToEnd();

  pir::Parameter* GetParameter(const std::string& name) const;

  void SetParameter(const std::string& name,
                    std::unique_ptr<pir::Parameter>&& parameter);

  const std::shared_ptr<pir::Builder>& GetBuilder() const { return builder_; }

  const pir::InsertionPoint& GetCurrentInsertionPoint() const {
    return builder_->insertion_point();
  }

  /// Set the insertion point to the specified insertion_point.
  void SetInsertionPoint(const pir::InsertionPoint& insertion_point) {
    builder_->set_insertion_point(insertion_point);
  }

  /// Set the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void SetInsertionPoint(pir::Operation* op) {
    builder_->set_insertion_point(op);
  }

  void SetInsertionPointAfter(pir::Operation* op) {
    builder_->SetInsertionPointAfter(op);
  }
  /// Set the insertion point to the end of specified block.
  void SetInsertionPointToBlockEnd(pir::Block* block) {
    builder_->SetInsertionPointToBlockEnd(block);
  }

  // push current insertion point to the stack.
  void PushInsertionPoint() {
    insertion_point_stack_.push(builder_->insertion_point());
  }
  // pop the insertion point and set it to the current insertion point.
  void LoadInsertionPoint();

  void SetOpRole(int op_role) { builder_->set_op_role(op_role); }
  int GetOpRole() const { return builder_->op_role(); }

  void SetChunkId(int chunk_id) { builder_->set_chunk_id(chunk_id); }
  int GetChunkId() const { return builder_->chunk_id(); }

 private:
  ApiBuilder();

  DISABLE_COPY_AND_ASSIGN(ApiBuilder);

  pir::IrContext* ctx_;
  std::shared_ptr<pir::Builder> builder_;
  std::stack<pir::InsertionPoint> insertion_point_stack_;
};

}  // namespace dialect
}  // namespace paddle
