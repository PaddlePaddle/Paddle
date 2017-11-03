/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

class RankSortLoDOp : public framework::OperatorBase {
 public:
  RankSortLoDOp(const std::string& type,
                const framework::VariableNameMap& inputs,
                const framework::VariableNameMap& outputs,
                const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  RankSortLoDOp(const RankSortLoDOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;
};

}  // namespace operators
}  // namespace paddle
