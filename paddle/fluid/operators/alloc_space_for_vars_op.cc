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

#include <vector>
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

class AllocSpaceForVarsOpOp : public framework::OperatorBase {
 public:
  AllocSpaceForVarsOpOp(const std::string &type,
                        const framework::VariableNameMap &inputs,
                        const framework::VariableNameMap &outputs,
                        const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &in_var_names = Inputs("Input");
    PADDLE_ENFORCE_GT(in_var_names.size(), 0);
    size_t mem_size = 0;
    framework::proto::VarType::Type fuse_space_type =
        static_cast<framework::proto::VarType::Type>(0);
    for (auto &name : in_var_names) {
      auto var = scope.FindVar(name);
      PADDLE_ENFORCE_NOT_NULL(var);
      // Only support LoDTensor,
      bool valid_var = var->IsType<framework::LoDTensor>();
      PADDLE_ENFORCE(valid_var, "");
      auto tensor = var->Get<framework::LoDTensor>();
      auto dtype = tensor.type();
      if (fuse_space_type == static_cast<framework::proto::VarType::Type>(0)) {
        fuse_space_type = dtype;
        PADDLE_ENFORCE_NE(dtype,
                          static_cast<framework::proto::VarType::Type>(0));
      }
      PADDLE_ENFORCE_EQ(dtype, fuse_space_type);
      auto size = tensor.numel();
      PADDLE_ENFORCE_GT(size, 0);
      mem_size += size;
    }
    auto out_var_names = Outputs("Input");

    PADDLE_ENFORCE_EQ(in_var_names.size(), out_var_names.size());
    auto out_tensor =
        scope.FindVar(out_var_names[0])->GetMutable<framework::LoDTensor>();
    auto origin_dim = out_tensor->dims();
    auto offset = framework::product(origin_dim);

    out_tensor->Resize(framework::make_ddim({static_cast<int64_t>(mem_size)}));
    out_tensor->mutable_data(dev_place, fuse_space_type);

    for (size_t i = 1; i < out_var_names.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_var_names[i], out_var_names[i]);
      auto out_t =
          scope.FindVar(out_var_names[i])->GetMutable<framework::LoDTensor>();
      auto &origin_dim = out_t->dims();
      int64_t len = out_t->numel();
      out_t->ShareDataWith(out_tensor->Slice(offset, offset + len));
      offset += len;
      out_t->Resize(origin_dim);
    }
    out_tensor->Resize(origin_dim);
  }
};

class AllocSpaceForVarsOpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "A set of variables.").AsDuplicable();
    AddOutput("Output", "A set of variables.").AsDuplicable();
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(alloc_space_for_vars,
                  paddle::operators::AllocSpaceForVarsOpOp,
                  paddle::operators::AllocSpaceForVarsOpOpMaker);
