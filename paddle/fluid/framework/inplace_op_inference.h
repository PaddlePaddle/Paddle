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
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimize_helper.h"
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
    PADDLE_ENFORCE(!op_desc.InputNames().empty(),
                   "Op inputs must not be empty");
    PADDLE_ENFORCE(!op_desc.OutputNames().empty(),
                   "Op outputs must not be empty");
    auto x_name = op_desc.InputNames().at(0);
    auto out_name = op_desc.OutputNames().at(0);
    return std::unordered_map<std::string, std::string>{{x_name, out_name}};
  }
};

/*
  Gradient op. Inplace output use it's Input.
  For example, Input@Grad->Input reuse strategy.
 */
class GradOpInplaceInToOut : public InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const OpDesc& op_desc, bool use_cuda) const override {
    std::unordered_map<std::string, std::string> ret;
    std::unordered_set<std::string> output_names(op_desc.OutputNames().begin(),
                                                 op_desc.OutputNames().end());
    for (auto& input_name : op_desc.InputNames()) {
      if (output_names.count(GradVarName(input_name))) {
        ret.insert({input_name, GradVarName(input_name)});
      }
    }
    return ret;
  }
};

}  // namespace framework
}  // namespace paddle
