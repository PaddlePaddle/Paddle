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

#include "paddle/infrt/dialect/tensorrt/trt_graph_fuse_pass.h"

#include <glog/logging.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <list>
#include <unordered_set>
#include <vector>

#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"

namespace infrt {
namespace trt {

// merge the first&second graph op to a new graph op.
static void mergeTwoAdjacentGraphOp(mlir::OpBuilder &builder,  // NOLINT
                                    ::infrt::GraphOp first,
                                    ::infrt::GraphOp second) {
  // comput inputs and outputs
  ::llvm::SmallVector<mlir::Value, 4> inputs(first.getOperands()), outputs;
  for (mlir::Value input : second.getOperands()) {
    if (input.getDefiningOp() != first) {
      inputs.push_back(input);
    }
  }
  ::llvm::DenseMap<mlir::Value, unsigned int> op_output_mapping;
  for (mlir::Value output : first.getResults()) {
    for (mlir::Operation *user : output.getUsers()) {
      if (user != second && user->getParentOp() != second) {
        op_output_mapping[output] = outputs.size();
        outputs.push_back(output);
        break;
      }
    }
  }
  auto return_op = second.getBody()->getTerminator();
  outputs.append(return_op->getOperands().begin(),
                 return_op->getOperands().end());
  ::llvm::SmallVector<mlir::Type, 4> return_types;
  for (auto value : outputs) {
    return_types.push_back(value.getType());
  }

  // create the new graph op
  builder.setInsertionPoint(first);
  auto loc = first.getLoc();
  auto graph_op = builder.create<::infrt::GraphOp>(loc, return_types, inputs);
  mlir::Block *block = new mlir::Block;
  auto copy_range = second.getBody()->without_terminator();
  block->getOperations().splice(block->begin(),
                                second.getBody()->getOperations(),
                                copy_range.begin(),
                                copy_range.end());
  copy_range = first.getBody()->without_terminator();
  block->getOperations().splice(block->begin(),
                                first.getBody()->getOperations(),
                                copy_range.begin(),
                                copy_range.end());
  builder.setInsertionPointToEnd(block);
  builder.create<::infrt::ReturnOp>(loc, outputs);
  graph_op.body().push_back(block);

  // mapping the output
  unsigned int num_result = first.getNumResults();
  return_op = first.getBody()->getTerminator();
  for (unsigned int index = 0; index < num_result; ++index) {
    auto origin_value = first.getResult(index);
    if (op_output_mapping.find(origin_value) == op_output_mapping.end()) {
      origin_value.replaceAllUsesWith(return_op->getOperand(index));
    } else {
      auto inner_value = return_op->getOperand(index);
      auto outer_value = graph_op.getResult(op_output_mapping[origin_value]);
      while (!origin_value.use_empty()) {
        auto replace_value =
            origin_value.use_begin()->getOwner()->getParentOp() == graph_op
                ? inner_value
                : outer_value;
        origin_value.use_begin()->set(replace_value);
      }
    }
  }
  second.replaceAllUsesWith(
      graph_op.getResults().take_back(second.getNumResults()));
  // first.erase();
  // second.erase();
}

struct TRT_Graph_Fuse : public ::mlir::RewritePattern {
  explicit TRT_Graph_Fuse(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("infrt.graph", 1, context, {}) {}
  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation *op, ::mlir::PatternRewriter &rewriter) const override {
    auto graph_op = ::llvm::dyn_cast<::infrt::GraphOp>(op);
    ::infrt::GraphOp father_graph_op{nullptr};
    for (::mlir::Value input : graph_op.inputs()) {
      if (input.getDefiningOp()) {
        if (auto father_graph_op =
                ::llvm::dyn_cast<::infrt::GraphOp>(input.getDefiningOp())) {
          // check mergeable
          bool mergeable{true};
          for (::mlir::Value output : father_graph_op->getOpResults()) {
            for (auto user : output.getUsers()) {
              if (user->getName().getIdentifier().str() == "infrt.graph" &&
                  user != graph_op) {
                mergeable = false;
                break;
              }
            }
            if (!mergeable) break;
          }

          if (mergeable) {
            // for debug
            LOG(INFO) << ".......................................mergeable....."
                         "..................................";
            LOG(INFO) << "graph Op.name: "
                      << graph_op->getName().getIdentifier().str();
            for (auto xx : graph_op.inputs()) {
              std::string str;
              llvm::raw_string_ostream os(str);
              xx.print(os);
              LOG(INFO) << str;
            }
            LOG(INFO) << "father_graph Op.name: "
                      << father_graph_op->getName().getIdentifier().str();
            for (auto xx : father_graph_op.inputs()) {
              std::string str;
              llvm::raw_string_ostream os(str);
              xx.print(os);
              LOG(INFO) << str;
            }

            // mergeTwoAdjacentGraphOp(rewriter, father_graph_op, graph_op);
            // return ::mlir::success();
            break;
          }
        }
      }
    }

    return rewriter.notifyMatchFailure(
        op, [&](::mlir::Diagnostic &diag) { diag << "infrt.graph not match"; });
  }
};

void TRTGraphFusePass::runOnFunction() {
  ::mlir::RewritePatternSet patterns(&getContext());
  patterns.add<TRT_Graph_Fuse>(&getContext());

  if (::mlir::failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

}  // namespace trt
}  // namespace infrt
