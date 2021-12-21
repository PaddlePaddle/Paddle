// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/diagnostic_utils.h"

#include <string>

namespace infrt::dialect {

struct MyScopedDiagnosicHandler::Impl {
  Impl() : diag_stream_(diag_str_) {}

  // String stream to assemble the final error message.
  std::string diag_str_;
  llvm::raw_string_ostream diag_stream_;

  // A SourceMgr to use for the base handler class.
  llvm::SourceMgr source_mgr_;

  // Log detail information.
  bool log_info_{};
};

MyScopedDiagnosicHandler::MyScopedDiagnosicHandler(mlir::MLIRContext *ctx,
                                                   bool propagate)
    : mlir::SourceMgrDiagnosticHandler(
          impl_->source_mgr_, ctx, impl_->diag_stream_),
      impl_(new Impl) {
  setHandler([this](mlir::Diagnostic &diag) { return this->handler(&diag); });
}

mlir::LogicalResult MyScopedDiagnosicHandler::handler(mlir::Diagnostic *diag) {
  if (diag->getSeverity() != mlir::DiagnosticSeverity::Error &&
      !impl_->log_info_)
    return mlir::success();
  emitDiagnostic(*diag);
  impl_->diag_stream_.flush();
  return mlir::failure(true);
}

}  // namespace infrt::dialect
