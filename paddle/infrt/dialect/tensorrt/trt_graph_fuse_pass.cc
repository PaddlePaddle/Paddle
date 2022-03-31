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
namespace {
// ReverseDfs
// do reverse dfs. calls "func" to search when visit a node.
// The elements in 'source' can't be nullptr.
// Reference the function nameed "FlexibleDFS" but defined in:
// paddle/fluid/framework/ir/subgraph_detector.cc.

bool reverseDfs(std::vector<mlir::Operation *> source,
                const std::function<bool(const mlir::Operation *)> &func) {
  std::unordered_set<const mlir::Operation *> visited;
  while (!source.empty()) {
    auto node = source.back();
    source.pop_back();
    if (visited.count(node)) continue;
    visited.insert(node);
    if (func(node)) return true;
    auto values = node->getOperands();
    for (auto value : values) {
      // if the value is a block argument, the node is nullptr.
      mlir::Operation *node = value.getDefiningOp();
      if (node != nullptr && !visited.count(node)) {
        source.emplace_back(node);
      }
    }
  }
  return false;
}

// merge the first&second graph op to a new graph op.
void mergeTwoAdjacentGraphOp(mlir::OpBuilder &builder,  // NOLINT
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
  first.erase();
  second.erase();
}

// Topological sort the function op.
void topoSortBlock(mlir::Block &body) {  // NOLINT
  llvm::SetVector<mlir::Operation *> toSort;
  if (body.empty()) return;
  for (auto it = body.rbegin(); it != body.rend(); ++it) {
    toSort.insert(&*it);
  }
  llvm::SetVector<mlir::Operation *> result = mlir::topologicalSort(toSort);
  for (auto *op : result) {
    op->moveBefore(body.getTerminator());
  }
}

}  // namespace
/*
// Implementation of the trtGraphFusePass.
void TRTGraphFusePass::runOnFunction() {
  mlir::Block &body = getFunction().front();
  mlir::OpBuilder builder(&body, body.begin());
  bool changed = false;
  do {
    changed = false;
    for (auto &op : body) {
      ::infrt::GraphOp graph_op =
          ::llvm::dyn_cast_or_null<::infrt::GraphOp>(&op);
      if (nullptr == graph_op) continue;

      for (auto user_op : op.getUsers()) {
        ::infrt::GraphOp user_graph_op =
            ::llvm::dyn_cast_or_null<::infrt::GraphOp>(user_op);
        if (nullptr == user_graph_op) continue;
        // get all dst input nodes except src.
        std::vector<mlir::Operation *> source_nodes;
        for (auto operand : user_op->getOperands()) {
          auto input = operand.getDefiningOp();
          if (input != &op && input != nullptr) {
            source_nodes.push_back(input);
          }
        }
        // Reverse DFS from the source_nodes.
        if (!reverseDfs(source_nodes,
                        [&op](const mlir::Operation *n) { return n == &op; })) {
          mergeTwoAdjacentGraphOp(builder, graph_op, user_graph_op);
          changed = true;
          break;
        }
      }
      if (changed) break;
    }
  } while (changed);

  // TODO(wilber): Implement a toposort for efficiency.
  // topoSortBlock(body);
}
*/

struct TRT_Graph_Fuse : public ::mlir::RewritePattern {
  explicit TRT_Graph_Fuse(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("infrt.graph", 1, context, {}) {}
  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation *op, ::mlir::PatternRewriter &rewriter) const override {
    auto graph_op = ::llvm::dyn_cast<::infrt::GraphOp>(op);
    for (::mlir::Value input : graph_op.getOperands()) {
      std::string str;
      llvm::raw_string_ostream os(str);
      input.print(os);
      LOG(INFO) << "cur op   ==>   " << str << "\n";
    }
    /*
    ::mlir::SmallVector<::mlir::Value, 4> operands;
    ::mlir::Operation::operand_range Input = casted_op.getODSOperands(0);
    ::mlir::Operation::operand_range Scale = casted_op.getODSOperands(1);
    ::mlir::Operation::operand_range Bias = casted_op.getODSOperands(2);

    // TODO(weishengying) : recompute this via params
    operands.push_back((*Input.begin()));
    operands.push_back((*Scale.begin()));
    operands.push_back((*Bias.begin()));
    operands.push_back((*Bias.begin()));

    trt::ScaleNdOp scaleNd_op;
    // inputs
    ::mlir::SmallVector<::mlir::Value, 4> trt_inputs;
    for (auto v : operands) {
      trt_inputs.push_back(v);
    }

    // resultTypes
    ::mlir::SmallVector<::mlir::Type, 4> resultTypes;
    for (auto v : casted_op.getODSResults(0)) {
      resultTypes.push_back(v.getType());
    }

    // attributes
    ::mlir::SmallVector<::mlir::NamedAttribute, 8> attributes;
    {
      auto mode_attr = rewriter.getI32IntegerAttr(1);
      attributes.emplace_back(rewriter.getStringAttr("mode"), mode_attr);
    }

    {
      auto axis_attr = rewriter.getI32IntegerAttr(-1);
      attributes.emplace_back(rewriter.getStringAttr("axis"), axis_attr);
    }
    auto result = rewriter
                      .create<trt::ScaleNdOp>(
                          op->getLoc(), resultTypes, operands, attributes)
                      .getODSResults(0);
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    // TODO(weishengying) : update it
    for (uint32_t i = 0; i < casted_op.getNumResults(); i++) {
      for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{result}) {
        tblgen_repl_values.push_back(v);
      }
    }
    rewriter.replaceOp(op, tblgen_repl_values);
    */
    // return ::mlir::success();
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
