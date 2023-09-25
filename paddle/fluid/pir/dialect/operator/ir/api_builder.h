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

#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/macros.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"

namespace paddle {
namespace dialect {
///
/// \brief APIBuilder is used in IR API for building op
///
class APIBuilder {
 public:
  static APIBuilder& Instance() {
    static APIBuilder api_builder;
    return api_builder;
  }
  void SetProgram(pir::Program* program);

  /// Set the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void SetInsertionPoint(pir::Operation* op);

  void ResetInsertionPointToStart();

  void ResetInsertionPointToEnd();

  pir::Parameter* GetParameter(const std::string& name) const;

  void SetParameter(const std::string& name,
                    std::unique_ptr<pir::Parameter>&& parameter);

  std::shared_ptr<pir::Builder> GetBuilder() { return builder_; }

 private:
  APIBuilder();

  DISABLE_COPY_AND_ASSIGN(APIBuilder);

  pir::IrContext* ctx_;
  std::shared_ptr<pir::Builder> builder_;
};

}  // namespace dialect
}  // namespace paddle
