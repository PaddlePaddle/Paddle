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
#include <glog/logging.h>
#include "paddle/fluid/inference/openvino/engine.h"

namespace paddle {
namespace inference {
namespace openvino {
template <typename T>
void OpenVINOEngine::BindingInput(const std::string& input_name,
                                  ov::element::Type ov_type,
                                  std::vector<size_t> data_shape,
                                  T* data,
                                  int64_t data_num,
                                  int64_t index) {
  ov::Output<const ov::Node> model_input =
      HaveInputTensorName(input_name) ? complied_model_.input(input_name)
                                      : complied_model_.input(index);

  PADDLE_ENFORCE_EQ(
      model_input.get_element_type() == ov_type,
      true,
      common::errors::PreconditionNotMet(
          "Model input_name[%s]  index[%d] requires input type [%s],but "
          "receives type [%s]",
          input_name,
          index,
          OVType2PhiType(model_input.get_element_type()),
          OVType2PhiType(ov_type)));

  if (IsModelStatic()) {
    PADDLE_ENFORCE_EQ(
        model_input.get_partial_shape().compatible(
            ov::PartialShape(data_shape)),
        true,
        common::errors::PreconditionNotMet(
            "Model input_name[%s] index[%d] requires input_shape is [%s]!",
            input_name,
            index,
            model_input.get_partial_shape().to_string()));
  }

  try {
    auto requestTensor = infer_request_.get_tensor(input_name);
    requestTensor.set_shape(data_shape);
    auto input_shape = requestTensor.get_shape();
    std::memcpy(
        requestTensor.data(), static_cast<void*>(data), data_num * sizeof(T));
    infer_request_.set_tensor(input_name, requestTensor);
  } catch (const std::exception& exp) {
    LOG(ERROR) << exp.what();
  }
}
}  // namespace openvino
}  // namespace inference
}  // namespace paddle
