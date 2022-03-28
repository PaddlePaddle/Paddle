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

#include "paddle/infrt/dialect/tensorrt/trt_type_convert_pass.h"

#include <glog/logging.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "paddle/infrt/dialect/infrt/common/types.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace {

class TrtTypeConvertPass
    : public mlir::PassWrapper<TrtTypeConvertPass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "TrtTypeConvertPass"; }

  void runOnFunction() override;
};

void TrtTypeConvertPass::runOnFunction() {
  mlir::Block& body = getFunction().front();
  auto* mlir_ctx = getFunction()->getContext();
  mlir::OpBuilder builder(&body, body.begin());

  std::vector<mlir::Operation*> worklist;
  mlir::Operation* ctx_op{nullptr};
  worklist.reserve(body.getOperations().size());
  for (auto& op : body) {
    worklist.push_back(&op);
    if (op.getName().getStringRef() == "phi_dt.create_context.gpu") {
      ctx_op = &op;
    }
  }

  ::infrt::LayoutType layout = ::infrt::LayoutType::NCHW;
  ::infrt::TargetType target = ::infrt::TargetType::GPU;
  for (auto& op : worklist) {
    if (auto tensor_map_get_op =
            llvm::dyn_cast<::infrt::phi::TensorMapGetTensorOp>(op)) {
      auto res = tensor_map_get_op.output();
      if (auto t = res.getType().dyn_cast<::infrt::DenseTensorType>()) {
        auto replace_type = ::infrt::DenseTensorType::get(
            mlir_ctx, t.getTarget(), t.getPrecision(), layout);
        res.setType(replace_type);
      }
    }
    if (auto create_engine = llvm::dyn_cast<::infrt::trt::CreateEngineOp>(op)) {
      // Insert `infrt.gpu.memcpy` op.
      for (auto arg : create_engine.getOperands()) {
        if (mlir::Operation* producer = arg.getDefiningOp()) {
          if (arg.getType().isa<::infrt::DenseTensorType>()) {
            builder.setInsertionPointAfter(producer);
            auto t = arg.getType().dyn_cast<::infrt::DenseTensorType>();
            if (producer->getName().getStringRef() !=
                    "phi_dt.tensor_map_get_tensor" &&
                t.getTarget() != ::infrt::TargetType::GPU) {
              auto replace_type = ::infrt::DenseTensorType::get(
                  mlir_ctx, target, t.getPrecision(), layout);
              CHECK_NOTNULL(ctx_op);
              auto mem_cpy_op = builder.create<::infrt::phi::GpuMemCopyOp>(
                  arg.getLoc(),
                  replace_type,
                  arg,
                  llvm::dyn_cast<::infrt::phi::CreateGPUContextOp>(ctx_op)
                      .output(),
                  mlir::BoolAttr::get(mlir_ctx, /*d2h*/ false));
              arg.replaceAllUsesExcept(mem_cpy_op.output(), mem_cpy_op);
            }
          }
        } else {
          auto blockArg = arg.cast<mlir::BlockArgument>();
          if (arg.getType().isa<::infrt::DenseTensorType>()) {
            auto t = arg.getType().dyn_cast<::infrt::DenseTensorType>();
            builder.setInsertionPointAfter(ctx_op);
            auto replace_type = ::infrt::DenseTensorType::get(
                mlir_ctx, ::infrt::TargetType::GPU, t.getPrecision(), layout);
            CHECK_NOTNULL(ctx_op);
            auto mem_cpy_op = builder.create<::infrt::phi::GpuMemCopyOp>(
                blockArg.getLoc(),
                replace_type,
                blockArg,
                llvm::dyn_cast<::infrt::phi::CreateGPUContextOp>(ctx_op)
                    .output(),
                mlir::BoolAttr::get(mlir_ctx, /*d2h*/ false));
            arg.replaceAllUsesExcept(mem_cpy_op.output(), mem_cpy_op);
          }
        }
      }

      // Change ops(in block) types.
      auto& block = create_engine.getRegion().getBlocks().front();
      for (auto& op : block.without_terminator()) {
        for (size_t i = 0; i < op.getNumResults(); ++i) {
          if (auto t = op.getResult(i)
                           .getType()
                           .dyn_cast<::infrt::DenseTensorType>()) {
            auto replace_type = ::infrt::DenseTensorType::get(
                mlir_ctx, ::infrt::TargetType::GPU, t.getPrecision(), layout);
            op.getResult(i).setType(replace_type);
          }
        }
      }
    } else if (auto list_get_tensor_op =
                   llvm::dyn_cast<::infrt::dt::TensorListGetTensorOp>(op)) {
      auto result = list_get_tensor_op.output();
      if (auto t = result.getType().dyn_cast<::infrt::DenseTensorType>()) {
        result.setType(::infrt::DenseTensorType::get(
            mlir_ctx, ::infrt::TargetType::GPU, t.getPrecision(), layout));
      }
    } else if (auto return_op = llvm::dyn_cast<::infrt::ReturnOp>(op)) {
      for (auto arg : return_op->getOperands()) {
        if (auto t = arg.getType().dyn_cast<::infrt::DenseTensorType>()) {
          if (t.getLayout() != ::infrt::LayoutType::ANY ||
              t.getTarget() != ::infrt::TargetType::CPU ||
              t.getPrecision() != ::infrt::PrecisionType::FLOAT32) {
            builder.setInsertionPoint(return_op);
            CHECK_NOTNULL(ctx_op);
            auto mem_cpy_op = builder.create<::infrt::phi::GpuMemCopyOp>(
                return_op.getLoc(),
                ::infrt::DenseTensorType::get(mlir_ctx,
                                              ::infrt::TargetType::CPU,
                                              t.getPrecision(),
                                              ::infrt::LayoutType::ANY),
                arg,
                llvm::dyn_cast<::infrt::phi::CreateGPUContextOp>(ctx_op)
                    .output(),
                mlir::BoolAttr::get(mlir_ctx, /*d2h*/ true));
            arg.replaceAllUsesExcept(mem_cpy_op.output(), mem_cpy_op);
          }
        }
      }
    }
  }
}

}  // namespace

namespace infrt {
namespace trt {

std::unique_ptr<mlir::Pass> createTrtTypeConvertPass() {
  return std::make_unique<TrtTypeConvertPass>();
}

}  // namespace trt
}  // namespace infrt
