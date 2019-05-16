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

#pragma once

#include <map>
#include <string>
#include "paddle/fluid/inference/anakin/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace anakin {

class ActivationOpConverter : public AnakinOpConverter {
 public:
  explicit ActivationOpConverter(const std::string &op_type);

  virtual void operator()(const framework::proto::OpDesc &op,
                          const framework::BlockDesc &block_desc,
                          const framework::Scope &scope,
                          bool test_mode) override;
  virtual ~ActivationOpConverter() {}

 private:
  std::string op_type_;
  std::string anakin_op_type_;
  std::map<std::string, std::string> anakin_op_types_{{"tanh", "TanH"},
                                                      {"sigmoid", "Sigmoid"}};
};

class TanhOpConverter : public ActivationOpConverter {
 public:
  TanhOpConverter() : ActivationOpConverter("tanh") {}
};

class SigmoidOpConverter : public ActivationOpConverter {
 public:
  SigmoidOpConverter() : ActivationOpConverter("sigmoid") {}
};
}  // namespace anakin
}  // namespace inference
}  // namespace paddle
