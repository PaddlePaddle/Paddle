/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/convert.h"

namespace paddle {
namespace inference {
namespace tensorrt {

void TensorRTConverter::ConvertOp(const framework::OpDesc& op) {
  std::string type = op.Type();
  PADDLE_ENFORCE(op_registry_.count(type), "No converter registered for op: %s",
                 type);
  std::function<void(const framework::OpDesc&)> op_converter =
      op_registry_.at(type);
  op_converter(op);
}

void TensorRTConverter::ConvertBlock(const framework::BlockDesc& block) {
  for (auto op : block.AllOps()) {
    ConvertOp(*op);
  }
}

void TensorRTConverter::RegisterOpConverters() {
  op_registry_["mul"] = ConvertMul;
  op_registry_["conv2d"] = ConvertConv2D;
}

void TensorRTConverter::ConvertMul(const framework::OpDesc& op) {
  LOG(INFO) << "convert a fluid mul op to tensorrt fc layer without bias";
}

void TensorRTConverter::ConvertConv2D(const framework::OpDesc& op) {
  LOG(INFO) << "convert a fluid Conv2d op to tensorrt conv layer without bias";
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
