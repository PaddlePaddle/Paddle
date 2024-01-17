// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/trans_buffer_with_dynamic_shape.h"

#include <numeric>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/dev_info_manager.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/string.h"

namespace cinn::optim {

namespace {

common::cas_intervals_t var_intervals = {};
cinn::common::SymbolicExprAnalyzer analyzer(var_intervals);

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  Mutator() : shared_mem_size_used_(0) {}

  void Visit(const ir::_Tensor_* tensor, Expr* expr) override {
    if (!tensor->buffer.defined()) return;
    auto buf = tensor->buffer.As<ir::_Buffer_>();
    auto buf_size = ir::Expr(1);

    for (size_t i = 0; i < expr->as_tensor()->shape.size(); ++i) {
      auto e = expr->as_tensor()->shape[i];
      auto buf_e = buf->shape[i];
      CHECK(ir::ir_utils::IRCompare(e, buf_e));
      if (buf->memory_type == ir::MemoryType::GPULocal) {
        e = cinn::common::AutoSimplify(e);
        if (!e.is_constant()) {
          auto new_shape = ir::ir_utils::IRCopy(e);
          new_shape = analyzer.UpperBound(new_shape);
          CHECK(new_shape.is_constant());
          e = new_shape;
        }
        buf_e = ir::ir_utils::IRCopy(e);
      }
      buf_size = buf_size * buf_e;
      Visit(&e, &e);
    }
    if (buf->memory_type == ir::MemoryType::GPUShared) {
      buf_size = analyzer.UpperBound(buf_size);
      CHECK(buf_size.is_constant());
      shared_mem_size_used_ += buf_size.get_constant();
    }
  }

  size_t shared_mem_size_used_;
};

}  // namespace

void TransBufferWithDynamicShape(ir::Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
#ifdef CINN_WITH_CUDA
  auto cur_dev_info =
      common::DevInfoMgr<common::Target::Arch::NVGPU>::GetDevInfo(0);
  size_t max_shm_per_block = cur_dev_info->GetMaxSharedMemPerBlock();
  CHECK(mutator.shared_mem_size_used_ <= max_shm_per_block)
      << "The shared memory size used by current kernel "
      << "is greater than the max shared memory per block";
#endif
}

}  // namespace cinn::optim
