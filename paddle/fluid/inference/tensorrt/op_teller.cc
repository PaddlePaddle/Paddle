// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/op_teller.h"

namespace paddle {
namespace inference {
namespace tensorrt {

// Just tell by the op_types.
struct SimpleOpTypeSetTeller : public Teller {
  SimpleOpTypeSetTeller() {
#if IS_TRT_VERSION_GE(5130)
    teller_set.insert("relu6");
#endif
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& desc) override {
    return teller_set.count(op_type);
  }

 private:
  std::unordered_set<std::string> teller_set{{"mul",
                                              "conv2d",
                                              "pool2d",
                                              "relu",
                                              "softmax",
                                              "sigmoid",
                                              "depthwise_conv2d",
                                              "batch_norm",
                                              "concat",
                                              "tanh",
                                              "pad",
                                              "elementwise_add",
                                              "elementwise_mul",
                                              "dropout",
                                              "prelu",
                                              "conv2d_transpose",
                                              "leaky_relu",
                                              "fc",
                                              "shuffle_channel",
                                              "swish",
                                              "split"}};
};

bool OpTeller::Tell(const std::string& op_type, const framework::OpDesc& desc) {
  for (auto& teller : tellers_) {
    if ((*teller)(op_type, desc)) return true;
  }
  return false;
}

OpTeller::OpTeller() { tellers_.emplace_back(new SimpleOpTypeSetTeller); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
