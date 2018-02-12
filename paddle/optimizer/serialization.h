//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include "OptimizerConfig.pb.h"
#include "paddle/utils/Logging.h"
#include "tensor.h"

namespace paddle {
namespace optimizer {

static void TensorToProto(const Tensor& tensor, TensorProto* proto) {
  proto->set_data_type(TensorProto::PADDLE_ELEMENT_TYPE_FLOAT32);
  std::stringstream os;
  for (size_t i = 0; i < tensor.size(); ++i) {
    os << tensor[i];
    proto->add_content(os.str());
    os.str(std::string());
  }
}

static void ProtoToTensor(const TensorProto& proto, Tensor* tensor) {
  std::stringstream sin;
  for (auto i = 0; i < proto.content_size(); ++i) {
    sin << proto.content(i);
    sin >> (*tensor)[i];
    sin.str(std::string());
    sin.clear();
  }
}

}  // namespace optimizer
}  // namespace paddle
