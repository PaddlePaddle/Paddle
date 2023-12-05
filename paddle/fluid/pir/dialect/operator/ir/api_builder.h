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
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"

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

  /// Set the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void set_insertion_point(pir::Operation* op);

  void ResetInsertionPointToStart();

  void ResetInsertionPointToEnd();

  pir::Parameter* GetParameter(const std::string& name) const;

  void SetParameter(const std::string& name,
                    std::unique_ptr<pir::Parameter>&& parameter);

  std::shared_ptr<pir::Builder> GetBuilder() { return builder_; }

  const pir::InsertionPoint& insertion_point() const {
    return builder_->insertion_point();
  }

  void set_insertion_point(const pir::InsertionPoint& insertion_point) {
    builder_->set_insertion_point(insertion_point);
  }
  void PushInsertionPoint(const pir::InsertionPoint& insertion_point);
  void PopInsertionPoint();

 private:
  ApiBuilder();

  DISABLE_COPY_AND_ASSIGN(ApiBuilder);

  pir::IrContext* ctx_;
  std::shared_ptr<pir::Builder> builder_;
  std::stack<pir::InsertionPoint> insertion_point_stack_;
};

}  // namespace dialect
}  // namespace paddle
