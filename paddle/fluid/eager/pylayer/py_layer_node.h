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

#pragma once

#include <Python.h>

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_meta.h"

namespace egr {

class GradNodePyLayer : public GradNodeBase {
 public:
  GradNodePyLayer(PyObject* ctx,
                  size_t bwd_in_slot_num,
                  size_t bwd_out_slot_num)
      : GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    ctx_ = ctx;
    Py_INCREF(ctx_);
  }

  ~GradNodePyLayer() override { Py_XDECREF(ctx_); };

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override { VLOG(6) << "Do nothing here now"; }

  std::string name() override {
    return "GradNodePyLayer_" + std::string(Py_TYPE(ctx_)->tp_name);
  }

  void SaveForwardOutputsMeta(
      const std::vector<std::vector<paddle::experimental::Tensor*>>&
          outputs_tensor) {
    forward_outputs_meta_.resize(outputs_tensor.size());
    forward_outputs_place_.resize(outputs_tensor.size());
    for (size_t i = 0; i < outputs_tensor.size(); i++) {
      forward_outputs_meta_[i].reserve(outputs_tensor[i].size());
      forward_outputs_place_[i].reserve(outputs_tensor[i].size());
      for (auto tensor : outputs_tensor[i]) {
        if (tensor->is_dense_tensor()) {
          forward_outputs_meta_[i].push_back(
              static_cast<phi::DenseTensor*>(tensor->impl().get())->meta());
        } else {
          forward_outputs_meta_[i].emplace_back();
        }
        forward_outputs_place_[i].emplace_back(tensor->place());
      }
    }
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GradNodePyLayer>(new GradNodePyLayer(*this));
    return copied_node;
  }

 private:
  PyObject* ctx_{nullptr};
  std::vector<std::vector<phi::DenseTensorMeta>> forward_outputs_meta_;
  std::vector<std::vector<paddle::platform::Place>> forward_outputs_place_;
};

}  // namespace egr
