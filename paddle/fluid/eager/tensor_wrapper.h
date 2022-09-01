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

/**
 * We now still need TensorWrapper and it is designed to Copy
 * tensor in autograd mode.
 *
 * Since in autograd usage, we need to pass autograd_meta to
 * backward computation however in tensor interface add to much
 * autograd_related method is not a good choice.
 *
 * In TensorWrapper we will keep autograd info to backward, only
 * for input var, but for output var it will only copy autograd
 * with no grad **/

#pragma once
#include <Python.h>
#include "paddle/fluid/eager/autograd_meta.h"

namespace egr {
class TensorWrapper {
 public:
  TensorWrapper() = default;
  explicit TensorWrapper(const paddle::experimental::Tensor& tensor,
                         bool no_need_buffer = false);
  ~TensorWrapper();

  paddle::experimental::Tensor recover();

  paddle::experimental::Tensor get_intermidiate_tensor();

  void clear();

 private:
  void check_inplace_version();

 private:
  bool no_need_buffer_ = false;
  paddle::experimental::Tensor intermidiate_tensor_;
  std::weak_ptr<egr::GradNodeBase> weak_grad_node_;
  uint32_t inplace_version_snapshot_ = 0;
  PyObject* padked_tensor_info_{nullptr};
  PyObject* unpack_hook_{nullptr};
};
}  // namespace egr
