/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/shape_inference.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {

std::vector<DDim> InferShapeContext::GetReaderDims(
    const std::string &name) const {
  const std::vector<std::string> &arg_names = Inputs(name);
  PADDLE_ENFORCE_EQ(
      arg_names.size(), 1UL,
      "Reader input '%s' should hold one element, but now it holds %d", name,
      arg_names.size());
  return this->GetRepeatedDims(arg_names[0]);
}

void InferShapeContext::SetReaderDims(const std::string &name,
                                      const std::vector<DDim> &dims) {
  const std::vector<std::string> &arg_names = Outputs(name);
  PADDLE_ENFORCE_EQ(
      arg_names.size(), 1UL,
      "Reader output '%s' should hold one element, but now it holds %d", name,
      arg_names.size());
  return this->SetRepeatedDims(arg_names[0], dims);
}

}  // namespace framework
}  // namespace paddle
