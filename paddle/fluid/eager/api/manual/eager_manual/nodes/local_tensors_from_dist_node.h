// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

class LocalTensorsFromDistGradNode : public egr::GradNodeBase {
 public:
  LocalTensorsFromDistGradNode() : egr::GradNodeBase() {
    VLOG(3) << " Construct LocalTensorsFromDistGrad Node.";
  }

  LocalTensorsFromDistGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(3) << " Construct LocalTensorsFromDistGrad Node, bwd_in_slot_num: "
            << bwd_in_slot_num << ", bwd_out_slot_num: " << bwd_out_slot_num;
  }

  ~LocalTensorsFromDistGradNode() override {
    VLOG(3) << " Destruct LocalTensorsFromDistGrad Node.";
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    input_.clear();
    SetIsTensorWrappersCleared(true);
  }

  std::string name() override { return "LocalTensorsFromDistGradNode"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node = std::shared_ptr<LocalTensorsFromDistGradNode>(
          new LocalTensorsFromDistGradNode(*this));
      return copied_node;
    }
  }

  // SetTensorWrapperX
  // Only input's meta is needed.
  void SetTensorWrapperNoNeedBuffer_Input(const paddle::Tensor& input) {
    input_ = egr::TensorWrapper(input, true);
  }
  void SetAttribute_global_mesh(const phi::distributed::ProcessMesh& mesh) {
    global_mesh_ = mesh;
  }
  void SetAttribute_global_placements(
      const phi::distributed::Placements& placements) {
    global_placements_ = placements;
  }
  void SetAttribute_local_index(int local_index) {
    local_tensor_idx_ = local_index;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  // Attributes
  int local_tensor_idx_;
  phi::distributed::ProcessMesh global_mesh_;
  phi::distributed::Placements global_placements_;
};
