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

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace framework {
namespace ir {

class SetReaderDeviceCountPass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const override;

 private:
  int GetDeviceCount() const;

  std::unordered_set<std::string> ReaderOpSet() const;

  const Scope *GlobalScope() const;
};

int SetReaderDeviceCountPass::GetDeviceCount() const {
  return static_cast<int>(
      Get<const std::vector<platform::Place>>(details::kPlaces).size());
}

std::unordered_set<std::string> SetReaderDeviceCountPass::ReaderOpSet() const {
  return {"create_py_reader"};
}

const Scope *SetReaderDeviceCountPass::GlobalScope() const {
  return Get<const std::vector<Scope *>>(details::kLocalScopes)[0];
}

void SetReaderDeviceCountPass::ApplyImpl(Graph *graph) const {
  auto dev_cnt = GetDeviceCount();
  auto reader_ops = ReaderOpSet();
  auto scope = GlobalScope();
  size_t found_op_num = 0;

  for (auto &node : graph->Nodes()) {
    if (node->IsOp() && node->Op() &&
        reader_ops.count(node->Op()->Type()) != 0) {
      auto &op_handle = dynamic_cast<details::ComputationOpHandle &>(
          node->Wrapper<details::OpHandleBase>());
      auto *op_desc = node->Op();
      auto &op_base_attrs =
          const_cast<framework::AttributeMap &>(op_handle.GetOp()->Attrs());
      int dev_idx = static_cast<int>(op_handle.GetScopeIdx());

      op_desc->SetAttr("device_index", dev_idx);
      op_desc->SetAttr("device_count", dev_cnt);

      op_base_attrs["device_index"] = dev_idx;
      op_base_attrs["device_count"] = dev_cnt;

      auto queue_name = op_handle.GetOp()->Input("blocking_queue");
      auto var = scope->FindVar(queue_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          platform::errors::NotFound("Blocking queue of DataLoader not found"));

      using QueueHolder =
          operators::reader::OrderedMultiDeviceLoDTensorBlockingQueueHolder;
      if (var->IsType<QueueHolder>()) {
        var->GetMutable<QueueHolder>()->GetQueue()->SetDeviceCount(dev_cnt);
      }

      ++found_op_num;
      VLOG(10) << "Found op " << op_desc->Type() << " on device " << dev_idx;
    }
  }

  VLOG(10) << "Found op number " << found_op_num;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(set_reader_device_count_pass,
              paddle::framework::ir::SetReaderDeviceCountPass)
    .RequirePassAttr(paddle::framework::details::kPlaces);
