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

namespace egr {

class GradNodePyLayer : public GradNodeBase {
 public:
  GradNodePyLayer(PyObject* ctx, size_t bwd_in_slot_num,
                  size_t bwd_out_slot_num)
      : GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    ctx_ = ctx;
  }

  ~GradNodePyLayer() override { Py_DECREF(ctx_); };

  // Functor: perform backward computations
  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      const std::vector<std::vector<paddle::experimental::Tensor>>& grads)
      override;

  std::string name() {
    return "GradNodePyLayer_" + std::string(Py_TYPE(ctx_)->tp_name);
  }

  /**
   * Register ReduceHook
   * **/
  void RegisterReduceHook(std::shared_ptr<TensorVoidHook>&& hook);

  /**
   * Apply ReduceHook here
   * **/
  inline bool ReduceHooksRegistered() { return reduce_hooks_.size() != 0; }
  void ApplyReduceHooks();

 private:
  std::weak_ptr<paddle::experimental::Tensor> weak_grad_;

  std::function<paddle::experimental::Tensor(
      const paddle::experimental::Tensor&)>
      retain_grad_hook_;

  std::vector<std::shared_ptr<TensorVoidHook>> reduce_hooks_;
  PyObject* ctx_{nullptr};
};

}  // namespace egr
