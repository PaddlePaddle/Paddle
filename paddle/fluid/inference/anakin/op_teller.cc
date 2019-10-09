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

#include "paddle/fluid/inference/anakin/op_teller.h"

namespace paddle {
namespace inference {
namespace anakin {

// Just tell by the op_types.
struct SimpleOpTypeSetTeller : public Teller {
  SimpleOpTypeSetTeller() {
    teller_set.insert("mul");
    teller_set.insert("fc");
    teller_set.insert("conv2d_fusion");
    teller_set.insert("split");
    teller_set.insert("relu");
    teller_set.insert("pool2d");
    teller_set.insert("elementwise_add");
    teller_set.insert("elementwise_mul");
    teller_set.insert("concat");
    teller_set.insert("tanh");
    teller_set.insert("conv2d");
    teller_set.insert("batch_norm");
    teller_set.insert("softmax");
    teller_set.insert("flatten2");
    teller_set.insert("reshape2");
    teller_set.insert("transpose2");
    teller_set.insert("density_prior_box");
    teller_set.insert("detection_out");
    teller_set.insert("dropout");
    teller_set.insert("sigmoid");
    teller_set.insert("sum");
    teller_set.insert("depthwise_conv2d");
    teller_set.insert("prior_box");
    teller_set.insert("leaky_relu");
    teller_set.insert("affine_channel");
    teller_set.insert("relu6");
    teller_set.insert("swish");
    teller_set.insert("shuffle_channel");
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& desc) override {
    return teller_set.count(op_type);
  }

 private:
  std::unordered_set<std::string> teller_set;
};

bool OpTeller::Tell(const std::string& op_type, const framework::OpDesc& desc) {
  for (auto& teller : tellers_) {
    if ((*teller)(op_type, desc)) return true;
  }
  return false;
}

OpTeller::OpTeller() { tellers_.emplace_back(new SimpleOpTypeSetTeller); }

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
