// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/reindex_transpose_buffer_pass.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/phi/core/enforce.h"

namespace cinn {
namespace optim {

using ir::stmt::Alloc;
using ir::stmt::BlockRef;
using ir::stmt::Evaluate;
using ir::stmt::For;
using ir::stmt::Free;
using ir::stmt::IfThenElse;
using ir::stmt::Let;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

namespace {

struct TransposeBufferInfo {
  std::set<ir::Buffer> buffers;
  int tile_size = 32;  // default to 32 (CUDA) for backward compatibility
};

TransposeBufferInfo CollectTransposeBuffers(const BlockRef& body) {
  TransposeBufferInfo info;

  const auto VisitFn = [&](const StmtRef& stmt) {
    if (!stmt.isa<Schedule>()) return;
    Schedule schedule = stmt.as<Schedule>();
    auto attr_it = schedule->attrs().find("transpose_stage");
    if (attr_it == schedule->attrs().end()) return;

    // The transpose_stage annotation has format "write:tile_size" or
    // "read:tile_size" (e.g. "write:64", "read:32"). Parse stage and
    // tile_size from this single annotation.
    std::string annotation = std::get<std::string>(attr_it->second);
    std::string stage;
    size_t colon_pos = annotation.find(':');
    if (colon_pos != std::string::npos) {
      stage = annotation.substr(0, colon_pos);
      info.tile_size = std::stoi(annotation.substr(colon_pos + 1));
    } else {
      // Backward compatibility: old format "write" / "read" without tile_size
      stage = annotation;
    }

    StmtRef store = schedule->body()->stmts().front();
    PADDLE_ENFORCE(
        store.isa<Store>(),
        ::common::errors::PreconditionNotMet(
            "The Schedule of transpose buffer must have a pure Store."));

    Store store_stmt = store.as<Store>();
    PADDLE_ENFORCE_NOT_NULL(
        store_stmt->value().As<ir::Load>(),
        ::common::errors::PreconditionNotMet(
            "The store value of transpose buffer must be a pure Load."));

    if (stage == "write") {
      ir::Buffer buffer = store_stmt->tensor().as_tensor()->buffer;
      info.buffers.insert(buffer);
    }
  };

  ir::stmt::Visit(body, VisitFn, [](auto) {});
  return info;
}

void ReplaceTransposeBuffersWithUnionBuffer(
    ir::LoweredFunc func,
    const std::set<ir::Buffer>& old_buffers,
    ir::Buffer new_buffer) {
  std::vector<ir::Buffer> new_temp_bufs;
  for (auto& buffer : func->temp_bufs) {
    if (old_buffers.count(buffer) > 0) continue;
    new_temp_bufs.push_back(buffer);
  }
  new_temp_bufs.push_back(new_buffer);
  func->temp_bufs = std::move(new_temp_bufs);
}

struct TransposeBufferIndicesMutator : public ir::stmt::StmtMutator<> {
  explicit TransposeBufferIndicesMutator(ir::Buffer union_buffer, int tile_size)
      : union_buffer_(union_buffer), tile_size_(tile_size) {}

  void operator()(BlockRef block) { VisitBlock(block); }

 private:
  void VisitStmt(Schedule stmt) override {
    Schedule schedule = stmt.as<Schedule>();
    auto attr_it = stmt->attrs().find("transpose_stage");
    if (attr_it == stmt->attrs().end()) {
      ir::stmt::StmtMutator<>::VisitBlock(stmt->body());
      return;
    }

    StmtRef store = stmt->body()->stmts().front();
    Store store_stmt = store.as<Store>();

    // Note: the shape of the transpose shared memory tile is
    // [tile_size, tile_size], where tile_size = warp_size (32 for CUDA,
    // 64 for some custom devices). The tile_size is encoded in the
    // "transpose_stage" annotation as "write:tile_size" or "read:tile_size".
    std::vector<ir::Expr> shape = {ir::Expr(tile_size_), ir::Expr(tile_size_)};

    // Parse the stage from annotation (format: "write:64" or "read:32")
    std::string annotation = std::get<std::string>(attr_it->second);
    std::string stage = annotation.substr(0, annotation.find(':'));

    if (stage == "write") {
      // at buffer write stage, re-index the store buffer
      store_stmt->set_indices({GetIndexY(), GetIndexX() ^ GetIndexY()});
      ir::Expr new_tensor = ir::ir_utils::IRCopy(store_stmt->tensor());
      new_tensor.as_tensor()->shape = shape;
      new_tensor.as_tensor()->buffer = union_buffer_;
      store_stmt->set_tensor(new_tensor);
    } else {
      // at buffer read stage, re-index the load buffer
      ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
      auto* load = new_value.As<ir::Load>();
      load->indices = {GetIndexX(), GetIndexY() ^ GetIndexX()};
      load->tensor.as_tensor()->shape = shape;
      load->tensor.as_tensor()->buffer = union_buffer_;
      store_stmt->set_value(new_value);
    }
  }

  ir::Expr GetIndexX() { return ir::Var("threadIdx.x"); }

  ir::Expr GetIndexY() {
    return inner_loop_var_ * ir::Expr(8) + ir::Var("threadIdx.y");
  }

  void VisitStmt(For stmt) override {
    if (stmt->is_serial()) {
      inner_loop_var_ = stmt->loop_var();
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(IfThenElse stmt) override {
    ir::stmt::BlockRef true_case = stmt->true_case();
    VisitBlock(true_case);
    stmt->set_true_case(true_case);
    if (stmt->false_case().defined()) {
      ir::stmt::BlockRef false_case = stmt->false_case();
      VisitBlock(false_case);
      stmt->set_false_case(false_case);
    }
  }

  void VisitStmt(Let stmt) override { return; }
  void VisitStmt(Store stmt) override { return; }
  void VisitStmt(Alloc stmt) override { return; }
  void VisitStmt(Free stmt) override { return; }
  void VisitStmt(Evaluate stmt) override { return; }

 private:
  ir::Buffer union_buffer_;
  ir::Var inner_loop_var_;
  int tile_size_;
};

}  // namespace

LogicalResult ReindexTransposeBufferPass::Run(ir::LoweredFunc func) {
  BlockRef body = func->body_block;

  // Step 1. Collect all transpose buffers in the function, also verify that the
  // transpose buffers are used properly.
  TransposeBufferInfo info = CollectTransposeBuffers(body);
  if (info.buffers.empty()) {
    return LogicalResult::success();
  }

  int tile_size = info.tile_size;

  // Step 2. Create a union buffer to replace all transpose buffers.
  // The union buffer's size is the size of the largest data type among the
  // transpose buffers multiplied by tile_size * tile_size (the tile area).
  int max_dtype_bytes = 0;
  for (auto& buffer : info.buffers) {
    max_dtype_bytes = std::max(max_dtype_bytes, buffer->dtype.bytes());
  }

  ir::Buffer union_buffer =
      ir::_Buffer_::Make("transpose_union_shm",
                         {ir::Expr(max_dtype_bytes * tile_size * tile_size)});
  union_buffer->dtype = common::UInt(8);
  union_buffer->memory_type = ir::MemoryType::GPUShared;

  ReplaceTransposeBuffersWithUnionBuffer(func, info.buffers, union_buffer);

  // Step 3. Swizzle the load & store indices of transpose buffers.
  TransposeBufferIndicesMutator mutator(union_buffer, tile_size);
  mutator(body);

  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateReindexTransposeBufferPass() {
  return std::make_unique<ReindexTransposeBufferPass>();
}

}  // namespace optim
}  // namespace cinn
