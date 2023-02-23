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

/// The code and design is mainly from mlir, very thanks to the great project.

#include <cassert>
#include <optional>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace infra {
namespace detail {
class AdaptorPass;
}  // namespace detail

/// We can access pass only from PassManager.
class Pass {
 public:
  virtual ~Pass() = default;

 protected:
  virtual void Run(mlir::Operation* op) = 0;

  virtual inline bool CanScheduleOn(mlir::Operation*) const { return true; }

  virtual mlir::LogicalResult Initialize(mlir::MLIRContext* context) {
    return mlir::success();
  }

  friend class PassManager;
  friend class detail::AdaptorPass;
};

// namespace detail {

// struct PassExecutionState {
//   PassExecutionState(mlir::Operation* ir) : ir(ir), pass_failed(false) {}

//   mlir::Operation* ir;
//   bool pass_failed;
// };

// } // namespace detail

// //
// class Pass {
//  public:
//   virtual ~Pass() = default;

//   std::optional<llvm::StringRef> getOpName() const { return op_name_; }

//   // //=== ----- ===//
//   // // Statistic
//   // //=== ----- ===//
//   // class Statistic : public llvm::Statistic {
//   //  public:
//   //   Statistic(Pass* owner, const char* name, const char* description);

//   //   Statistic& operator=(unsigned value);
//   // };

//   // llvm::ArrayRef<Statistic*> GetStatistics() const { return statistics_; }
//   // llvm::MutableArrayRef<Statistic*> GetStatistics() { return statistics_;
//   }
//  protected:
//   explicit Pass(std::optional<llvm::StringRef> op_name = std::nullopt)
//       : op_name_(op_name) {}

//   ///
//   detail::PassExecutionState& GetPassState() {
//     assert(pass_state_ && "pass state was never initialized");
//     return *pass_state_;
//   }

//   mlir::MLIRContext& GetContext() { return *GetOperation()->getContext(); }

//   virtual void RunOpOperation() = 0;

//   virtual bool CanScheduleOn(mlir::RegisteredOperationName op_name) const =
//   0;

//   mlir::Operation* GetOperation() {
//     return GetPassState().ir;
//   }

//   void SignalPassFailure() { GetPassState().pass_failed = true; }

//  private:

//   /// The name of the operation that this pass operates on, or std::nullopt
//   if this is a generic OperationPass. std::optional<llvm::StringRef>
//   op_name_;

//   /// The current execution state for the pass.
//   std::optional<detail::PassExecutionState> pass_state_;

//   // /// The set of statistics held by this pass.
//   // std::vector<Statistic*> statistics_;

//   friend class PassManager;
// };

//==------==//
// Pass Model Definitions
//==------==//

/// Pass to transform an operation of a specified type.
///
///

// template <typename OpT = void>
// class OperationPass : public Pass {
//  protected:
//   OperationPass() : Pass(OpT::getOperationName()) {}

//   bool CanScheduleOn(mlir::RegisteredOperationName op_name) const final {
//     return op_name.getStringRef() == getOpName();
//   }

//   OpT GetOperation() { return llvm::cast<OpT>(Pass::GetOperation()); }
// };

// /// Pass to transform an operation.
// template<>
// class OperationPass<void> : public Pass {
//  protected:
//   OperationPass() : Pass() {}

//   bool CanScheduleOn(mlir::RegisteredOperationName op_name) const override {
//     return true;
//   }
// };

}  // namespace infra
