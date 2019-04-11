// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/modify_op_lock_and_record_event_pass.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_graph_view.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

static bool IsLockAndRecordEventFreeComputationOpHandle(
    ComputationOpHandle *op, const OpGraphView &graph_view) {
  if (!platform::is_gpu_place(op->GetPlace())) return false;
  for (auto &pending_op : graph_view.PendingOps(op)) {
    auto *tmp = dynamic_cast<ComputationOpHandle *>(pending_op);
    if (tmp == nullptr || !(tmp->GetPlace() == op->GetPlace())) {
      return false;
    }
  }
  return true;
}

void ModifyOpLockAndRecordEventPass::ApplyImpl(ir::Graph *ir_graph) const {
  auto all_ops = ir::FilterByNodeWrapper<OpHandleBase>(*ir_graph);
  OpGraphView graph_view(all_ops);
  for (auto &op : all_ops) {
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
    if (compute_op == nullptr) continue;
    bool is_lock_and_record_event_free =
        IsLockAndRecordEventFreeComputationOpHandle(compute_op, graph_view);
    compute_op->SetLockAndRecordEventFree(is_lock_and_record_event_free);
    if (is_lock_and_record_event_free) {
      VLOG(10) << "Set is_lock_and_record_event_free be true in op "
               << compute_op->DebugString();
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(modify_op_lock_and_record_event_pass,
              paddle::framework::details::ModifyOpLockAndRecordEventPass);
