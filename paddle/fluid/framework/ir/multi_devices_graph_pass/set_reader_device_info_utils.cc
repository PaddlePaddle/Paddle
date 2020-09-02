// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/multi_devices_graph_pass/set_reader_device_info_utils.h"
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace framework {
namespace ir {

static std::unordered_set<std::string> ReaderOpSet() {
  return {"create_py_reader"};
}

void InitReaderQueueDeviceCount(Graph *graph, const Scope &scope,
                                size_t dev_cnt) {
  using QueueHolder =
      operators::reader::OrderedMultiDeviceLoDTensorBlockingQueueHolder;

  auto reader_ops = ReaderOpSet();
  for (auto &node : graph->Nodes()) {
    if (node->IsOp() && node->Op() &&
        reader_ops.count(node->Op()->Type()) != 0) {
      auto queue_name = node->Op()->Input("blocking_queue")[0];
      auto var = scope.FindVar(queue_name);
      if (var && var->IsType<QueueHolder>()) {
        VLOG(10) << "Set device count of " << queue_name << " to be "
                 << dev_cnt;
        var->GetMutable<QueueHolder>()->GetQueue()->SetDeviceCount(dev_cnt);
      }
    }
  }
}

void SetReaderOpDeviceInfo(Graph *graph, size_t dev_cnt, size_t dev_idx) {
  auto reader_ops = ReaderOpSet();
  size_t found_op_num = 0;

  for (auto &node : graph->Nodes()) {
    if (node->IsOp() && node->Op() &&
        reader_ops.count(node->Op()->Type()) != 0) {
      auto &op_handle = dynamic_cast<details::ComputationOpHandle &>(
          node->Wrapper<details::OpHandleBase>());
      auto *op_desc = node->Op();
      auto &op_base_attrs =
          const_cast<framework::AttributeMap &>(op_handle.GetOp()->Attrs());
      int actual_dev_idx = static_cast<int>(op_handle.GetScopeIdx());
      if (dev_idx != -1UL) {
        actual_dev_idx = static_cast<int>(dev_idx);
      }

      op_desc->SetAttr("device_index", actual_dev_idx);
      op_desc->SetAttr("device_count", static_cast<int>(dev_cnt));

      op_base_attrs["device_index"] = actual_dev_idx;
      op_base_attrs["device_count"] = static_cast<int>(dev_cnt);

      ++found_op_num;
      VLOG(10) << "Found op " << op_desc->Type() << " on device "
               << actual_dev_idx;
    }
  }

  VLOG(10) << "Found op number " << found_op_num;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
