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

#pragma once
#include <string>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace framework {

class VarTypeInference {
 public:
  virtual ~VarTypeInference() {}
  virtual void operator()(const OpDesc& op_desc, BlockDesc* block) const = 0;
};

class PassInDtypeAndVarTypeToOutput : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const final {
    auto in_out_var_names = this->GetInputOutputWithSameType();

    for (auto& i_o_n : in_out_var_names) {
      auto& x_name = op_desc.Input(i_o_n.first).at(0);
      auto& out_name = op_desc.Output(i_o_n.second).at(0);

      auto& x = block->FindRecursiveOrCreateVar(x_name);
      auto& out = block->FindRecursiveOrCreateVar(out_name);
      out.SetType(x.GetType());
      out.SetDataType(x.GetDataType());
    }
  }

 protected:
  virtual std::unordered_map<std::string, std::string>
  GetInputOutputWithSameType() const = 0;
};

}  // namespace framework
}  // namespace paddle
