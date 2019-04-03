// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/anakin/convert/dropout.h"
#include <algorithm>
#include <string>
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void DropoutOpConverter<TargetT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Mask").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto x_name = op_desc.Input("X").front();
  auto out_name = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "Scale", {x_name}, {out_name});

  auto dropout_prob = boost::get<float>(op_desc.GetAttr("dropout_prob"));
  auto factor = 1 - dropout_prob;
  Shape shape1(std::vector<int>({1, 1, 1, 1}));
  auto *weight1 =
      GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(shape1);
  auto *factor_data = static_cast<float *>(weight1->h_tensor().mutable_data());
  float weight1_data[] = {factor};
  std::copy(std::begin(weight1_data), std::end(weight1_data), factor_data);

  this->engine_->AddOpAttr(op_name, "weight_1", *weight1);
  this->engine_->AddOpAttr(op_name, "axis", 0);
  this->engine_->AddOpAttr(op_name, "num_axes", 0);
  this->engine_->AddOpAttr(op_name, "bias_term", false);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
REGISTER_CUDA_ANAKIN_OP_CONVERTER(dropout,
                                  DropoutOpConverter<::anakin::saber::NV>);
#endif
REGISTER_CPU_ANAKIN_OP_CONVERTER(dropout,
                                 DropoutOpConverter<::anakin::saber::X86>);
