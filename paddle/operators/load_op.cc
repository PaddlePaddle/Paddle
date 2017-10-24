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

#include "paddle/framework/op_registry.h"

#include <fstream>

namespace paddle {
namespace operators {

class LoadOp : public framework::OperatorBase {
 public:
  LoadOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto filename = Attr<std::string>("file_path");
    std::ifstream fin(filename);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s for load op",
                   filename);

    auto out_var_name = Output("Out");
    auto *out_var = scope.FindVar(out_var_name);
    PADDLE_ENFORCE(out_var != nullptr, "Output variable %s cannot be found",
                   out_var_name);

    auto *tensor = out_var->GetMutable<framework::LoDTensor>();

    uint32_t version;
    fin.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
    PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
    framework::TensorDesc desc;
    {  // int32_t size
      // protobuf
    }
  }
};

}  // namespace operators
}  // namespace paddle