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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {

static framework::proto::VarType::Type kDefaultDtype =
    static_cast<framework::proto::VarType::Type>(0);

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
    auto &param_var_names = Inputs("Parameters");
    auto &grad_var_names = Outputs("Gradients");
    PADDLE_ENFORCE_GT(param_var_names.size(), 0);
    PADDLE_ENFORCE_EQ(param_var_names.size(), grad_var_names.size());

    size_t mem_size = 0;
    framework::proto::VarType::Type dtype = kDefaultDtype;
    GetMemSizeAndDtype(scope, param_var_names, grad_var_names, &mem_size,
                       &dtype);

    auto out_tensor =
        scope.FindVar(grad_var_names[0])->GetMutable<framework::LoDTensor>();
    auto &origin_dim = out_tensor->dims();
    int64_t offset = framework::product(origin_dim);

    out_tensor->Resize(framework::make_ddim({static_cast<int64_t>(mem_size)}))
        .mutable_data(dev_place, dtype);

    VLOG(10) << "alloc_space_for_vars: output(" << grad_var_names[0]
             << ") ,dim:(" << origin_dim << ")"
             << " Address: " << out_tensor->data<void>();

    for (size_t i = 1; i < grad_var_names.size(); ++i) {
      auto out_t =
          scope.FindVar(grad_var_names[i])->GetMutable<framework::LoDTensor>();

      int64_t len = out_t->numel();
      out_t->ShareDataWith(out_tensor->Slice(offset, offset + len))
          .Resize(out_t->dims());
      offset += len;
      VLOG(10) << "alloc_space_for_vars: output(" << grad_var_names[i]
               << ") ,dim:(" << origin_dim << ")"
               << " Address: " << out_t->data<void>();
    }

    out_tensor->Resize(origin_dim);
  }

  void GetMemSizeAndDtype(const framework::Scope &scope,
                          const std::vector<std::string> &param_var_names,
                          const std::vector<std::string> &grad_var_names,
                          size_t *mem_size,
                          framework::proto::VarType::Type *dtype) const {
    *mem_size = 0;
    for (size_t i = 0; i < grad_var_names.size(); ++i) {
      auto grad_var = scope.FindVar(grad_var_names[i]);
      PADDLE_ENFORCE_NOT_NULL(grad_var, "%s", grad_var_names[i]);

      // Only support LoDTensor,
      PADDLE_ENFORCE(grad_var->IsType<framework::LoDTensor>(), "%s",
                     grad_var_names[i]);

      // Note: Assume that the dtype of parameter and gradient are the same.
      // Doesn't get dtype from grad_var in runtime.
      auto &p_tensor =
          scope.FindVar(param_var_names[i])->Get<framework::LoDTensor>();

      auto p_dtype = p_tensor.type();
      if (*dtype == kDefaultDtype) {
        PADDLE_ENFORCE_NE(p_dtype, kDefaultDtype, "%s", grad_var_names[i]);
        *dtype = p_dtype;
      }
      PADDLE_ENFORCE_EQ(p_dtype, *dtype, "%s", grad_var_names[i]);

      grad_var->GetMutable<framework::LoDTensor>()->Resize(p_tensor.dims());
      auto size = p_tensor.numel();
      VLOG(10) << "alloc_space_for_vars: input(" << grad_var_names[i]
               << ") ,dim:(" << p_tensor.dims() << ")";
      PADDLE_ENFORCE_GT(size, 0, "%s", grad_var_names[i]);
      *mem_size += size;
    }
  }
};

class AllocSpaceForVarsOpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Parameters", "A set of variables.").AsDuplicable();
    AddOutput("Gradients", "A set of variables.").AsDuplicable();
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(alloc_space_for_vars,
                  paddle::operators::AllocSpaceForVarsOpOp,
                  paddle::operators::AllocSpaceForVarsOpOpMaker);
