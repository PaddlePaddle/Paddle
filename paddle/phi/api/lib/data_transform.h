/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace experimental {

class TransformFlag {
 public:
  TransformFlag(bool stop_transform = false,
                bool trans_dtype = false,
                bool trans_backend = true,
                bool trans_layout = true)
      : stop_transform_(stop_transform),
        trans_data_type_(trans_dtype),
        trans_backend_(trans_backend),
        trans_layout_(trans_layout) {}

  bool NeedTransform() const {
    return !stop_transform_ &&
           (trans_data_type_ || trans_backend_ || trans_layout_);
  }

  bool need_trans_data_type() const {
    return !stop_transform_ && trans_data_type_;
  }

  bool need_trans_backend() const { return !stop_transform_ && trans_backend_; }

  bool need_trans_layout() const { return !stop_transform_ && trans_layout_; }

 private:
  // This is the highest priority in flags,
  // and can be setted by api[data_transform->skip_transform] in the yaml file.
  bool stop_transform_ = false;

  // trans_data_type_ can be setted by api[data_transform->support_trans_dtype]
  // in the yaml file.
  // trans_data_type_ only affect the non complex types,
  // the complex is always transferd, except stop_transform_ is true.
  bool trans_data_type_ = false;

  // trans_backend_ and trans_layout_ are true defalutly,
  // and they can only be setted by global flag.
  bool trans_backend_ = true;
  bool trans_layout_ = true;
};

std::shared_ptr<phi::DenseTensor> PrepareData(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

std::shared_ptr<phi::DenseTensor> PrepareData(
    const paddle::optional<Tensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

std::unique_ptr<std::vector<phi::DenseTensor>> PrepareData(
    const std::vector<Tensor>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

std::shared_ptr<phi::DenseTensor> PrepareData(
    const paddle::optional<const Tensor&> input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

}  // namespace experimental
}  // namespace paddle
