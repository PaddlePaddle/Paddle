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

#include <list>
#include <unordered_set>
#include <vector>
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Builders.h"
#include "paddle/infrt/dialect/pd_ops.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace infrt {
namespace trt {
namespace {

// FlexibleDFS
// do reverse dfs. calls leave(node) after visiting all parents of node.
// Reference the function with the same name but defined in:
// paddle/fluid/framework/ir/subgraph_detector.cc.
void FlexibleDFS(const std::vector<::mlir::Operation *> &source,
                 const std::function<bool(const ::mlir::Operation *)> &leave) {
  typedef struct {
    ::mlir::Operation *node;
    bool leave;
  } FNode;

  std::vector<FNode> stack;
  for (auto &node : source) {
    stack.push_back(FNode{node, false});
  }
  std::unordered_set<const ::mlir::Operation *> visited;
  while (!stack.empty()) {
    auto fnode = stack.back();
    stack.pop_back();

    if (fnode.leave) {
      if (leave && !leave(fnode.node)) return;
    }
    if (visited.count(fnode.node)) continue;
    visited.insert(fnode.node);

    if (leave) stack.push_back(FNode{fnode.node, true});
    auto values = fnode.node->getOperands();
    for (auto value : values) {
      ::mlir::Operation *node = value.getDefiningOp();
      if (!visited.count(node)) {
        stack.push_back(FNode{node, false});
      }
    }
  }
}

// merge the first&second graph op to a new graph op.
void mergeTwoAdjacentGraphOp(::mlir::OpBuilder &builder,  // NOLINT
                             ::mlir::pd::GraphOp first,
                             ::mlir::pd::GraphOp second) {
  // comput inputs and outputs
  ::llvm::SmallVector<::mlir::Value, 4> inputs(first.getOperands()), outputs;
  for (::mlir::Value input : second.getOperands()) {
    if (input.getDefiningOp() != first) {
      inputs.push_back(input);
    }
  }
  ::llvm::DenseMap<::mlir::Value, unsigned int> op_output_mapping;
  for (::mlir::Value output : first.getResults()) {
    for (::mlir::Operation *user : output.getUsers()) {
      if (user != second && user->getParentOp() != second) {
        op_output_mapping[output] = outputs.size();
        outputs.push_back(output);
        break;
      }
    }
  }
  auto fetch_op = second.getBody()->getTerminator();
  outputs.append(fetch_op->getOperands().begin(),
                 fetch_op->getOperands().end());
  ::llvm::SmallVector<::mlir::Type, 4> fetch_types;
  for (auto value : outputs) {
    fetch_types.push_back(value.getType());
  }

  // create the new graph op
  builder.setInsertionPoint(first);
  auto loc = first.getLoc();
  auto graph_op = builder.create<::mlir::pd::GraphOp>(loc, fetch_types, inputs);
  ::mlir::Block *block = new ::mlir::Block;
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
  builder.create<mlir::pd::FetchOp>(loc, outputs);
  graph_op.body().push_back(block);

  // mapping the output
  unsigned int num_result = first.getNumResults();
  fetch_op = first.getBody()->getTerminator();
  for (unsigned int index = 0; index < num_result; ++index) {
    auto origin_value = first.getResult(index);
    if (op_output_mapping.find(origin_value) == op_output_mapping.end()) {
      origin_value.replaceAllUsesWith(fetch_op->getOperand(index));
    } else {
      auto inner_value = fetch_op->getOperand(index);
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

}  // namespace

// Implementation of the trtGraphFusePass.
void trtGraphFusePass::runOnFunction() {
  mlir::Block &body = getFunction().front();
  ::mlir::OpBuilder builder(&body, body.begin());
  bool changed = false;
  do {
    changed = false;
    for (auto &op : body) {
      ::mlir::pd::GraphOp graph_op =
          ::llvm::dyn_cast_or_null<::mlir::pd::GraphOp>(&op);
      if (nullptr == graph_op) continue;

      for (auto user_op : op.getUsers()) {
        ::mlir::pd::GraphOp user_graph_op =
            ::llvm::dyn_cast_or_null<::mlir::pd::GraphOp>(user_op);
        if (nullptr == user_graph_op) continue;
        // get all dst input nodes except src.
        std::vector<::mlir::Operation *> source_nodes;
        for (auto operand : user_op->getOperands()) {
          auto input = operand.getDefiningOp();
          if (input != &op) {
            source_nodes.push_back(input);
          }
        }
        // Reverse DFS from the source_nodes.
        bool have_excess_path = false;
        FlexibleDFS(source_nodes,
                    [&have_excess_path, &op](const ::mlir::Operation *n) {
                      if (n == &op) {
                        have_excess_path = true;
                        return false;
                      }
                      return true;
                    });
        if (!have_excess_path) {
          mergeTwoAdjacentGraphOp(builder, graph_op, user_graph_op);
          changed = true;
          break;
        }
      }
      if (changed) break;
    }
  } while (changed);
}

}  // namespace trt
}  // namespace infrt
