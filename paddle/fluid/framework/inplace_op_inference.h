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
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace framework {

/*
  Inplace Inference for create In->Out pairs for inplaced operator.
  If we specify a pair of corresponding names. For example, X->Out.
  then Out will inplaced use X's memory. The base class will do
  legality validation for both variables.
*/

class InplaceOpInference {
 public:
  virtual ~InplaceOpInference() {}
  virtual std::unordered_map<std::string, std::string> operator()(
      const OpDesc& op_desc, bool use_cuda) const = 0;
};

/*
  Inplace In and Out for operator only have an Input and an Output.
  For example, activation op.
 */
class SingleOpInplaceInToOut : public InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const OpDesc& op_desc, bool use_cuda) const override {
    PADDLE_ENFORCE_EQ(op_desc.InputNames().size(), 1,
                      "Op inputs must be unique");
    PADDLE_ENFORCE_EQ(op_desc.OutputNames().size(), 1,
                      "Op outputs must be unique");
    auto x_name = op_desc.InputNames().at(0);
    auto out_name = op_desc.OutputNames().at(0);
    return std::unordered_map<std::string, std::string>{{x_name, out_name}};
  }
};

}  // namespace framework
}  // namespace paddle
